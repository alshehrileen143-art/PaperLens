"""Safety checks for uploaded audio (/ask_voice), run before it is sent for transcription.

Each format's duration is measured from the audio stream itself (MP3 frames,
M4A sample tables, WebM block timestamps, WAV data size), not from a header
field such as an MP3 Xing frame count or a WebM Duration element, which a file
could set to understate its length. Where two measures exist, the longer one
is used. Checks raise UploadRejected (see upload_validation.py).
"""

import struct
from pathlib import Path

from config import MAX_AUDIO_SECONDS
from upload_validation import UploadRejected

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".webm")


class _Unreadable(Exception):
    """The file has the right signature but its structure could not be parsed."""


# --- MP3 ---

# Bitrates in kbps, indexed by the header's 4-bit bitrate index (0 = free format, 15 = invalid).
_MP3_BITRATES = {
    (1, 1): (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    (1, 2): (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    (1, 3): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    (2, 1): (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    (2, 2): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    (2, 3): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_MP3_SAMPLE_RATES = {1: (44100, 48000, 32000), 2: (22050, 24000, 16000), 2.5: (11025, 12000, 8000)}
_MP3_VERSIONS = {0b00: 2.5, 0b10: 2, 0b11: 1}  # 0b01 is reserved.
_MP3_LAYERS = {0b01: 3, 0b10: 2, 0b11: 1}  # 0b00 is reserved.


def _mp3_frame(data: bytes, pos: int) -> tuple[int, float] | None:
    """(frame length in bytes, frame duration in seconds) if a valid frame header starts at pos."""
    if pos + 4 > len(data) or data[pos] != 0xFF or data[pos + 1] & 0xE0 != 0xE0:
        return None
    b1, b2 = data[pos + 1], data[pos + 2]
    version = _MP3_VERSIONS.get((b1 >> 3) & 0b11)
    layer = _MP3_LAYERS.get((b1 >> 1) & 0b11)
    bitrate_index, rate_index, padding = b2 >> 4, (b2 >> 2) & 0b11, (b2 >> 1) & 1
    if version is None or layer is None or bitrate_index in (0, 15) or rate_index == 3:
        return None
    bitrate = _MP3_BITRATES[(1 if version == 1 else 2, layer)][bitrate_index] * 1000
    sample_rate = _MP3_SAMPLE_RATES[version][rate_index]
    if layer == 1:
        return (12 * bitrate // sample_rate + padding) * 4, 384 / sample_rate
    samples = 1152 if layer == 2 or version == 1 else 576
    return samples // 8 * bitrate // sample_rate + padding, samples / sample_rate


def _mp3_seconds(data: bytes) -> float:
    pos = 0
    if data.startswith(b"ID3") and len(data) >= 10:
        size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]  # Syncsafe integer.
        pos = 10 + size + (10 if data[5] & 0x10 else 0)  # Footer flag.
    seconds, frames = 0.0, 0
    while pos < len(data):
        frame = _mp3_frame(data, pos)
        if frame is None:
            # Junk or a trailing tag: resync on the next possible header.
            next_pos = data.find(b"\xff", pos + 1)
            if next_pos == -1:
                break
            pos = next_pos
            continue
        length, duration = frame
        seconds += duration
        frames += 1
        pos += length
    if frames == 0:
        raise _Unreadable
    return seconds


# --- WAV ---

def _wav_seconds(data: bytes) -> float:
    pos, byte_rate, audio_bytes = 12, None, 0
    while pos + 8 <= len(data):
        chunk_id, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        if chunk_id == b"fmt " and size >= 16:
            byte_rate = struct.unpack_from("<I", data, body + 8)[0]
        elif chunk_id == b"data":
            # Streamed WAVs declare 0 or 0xFFFFFFFF; truncated ones declare too much.
            available = len(data) - body
            audio_bytes += available if size in (0, 0xFFFFFFFF) else min(size, available)
        pos = body + size + (size & 1)  # Chunks are padded to an even length.
    if not byte_rate or not audio_bytes:
        raise _Unreadable
    return audio_bytes / byte_rate


# --- M4A ---

_MP4_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def _mp4_boxes(data: bytes, start: int, end: int):
    """Yield (type, body start, body end) for each box in data[start:end]."""
    pos = start
    while pos + 8 <= end:
        size, box_type = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                raise _Unreadable
            size, header = struct.unpack_from(">Q", data, pos + 8)[0], 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise _Unreadable
        yield box_type, pos + header, pos + size
        pos += size


def _mp4_track_seconds(data: bytes, start: int, end: int) -> float | None:
    """Duration of an audio track ('trak' box body), or None if it is not audio."""
    handler, timescale, header_duration, sample_duration = None, None, 0, 0

    def walk(begin: int, finish: int) -> None:
        nonlocal handler, timescale, header_duration, sample_duration
        for box_type, body, body_end in _mp4_boxes(data, begin, finish):
            if box_type in _MP4_CONTAINERS:
                walk(body, body_end)
            elif box_type == b"hdlr" and body_end - body >= 12:
                handler = data[body + 8:body + 12]
            elif box_type == b"mdhd" and body_end - body >= 24:
                if data[body] == 1:  # Version 1: 64-bit times.
                    if body_end - body < 36:
                        raise _Unreadable
                    timescale, header_duration = struct.unpack_from(">IQ", data, body + 20)
                else:
                    timescale, header_duration = struct.unpack_from(">II", data, body + 12)
            elif box_type == b"stts" and body_end - body >= 8:
                count = struct.unpack_from(">I", data, body + 4)[0]
                if body + 8 + count * 8 > body_end:
                    raise _Unreadable
                for i in range(count):
                    samples, delta = struct.unpack_from(">II", data, body + 8 + i * 8)
                    sample_duration += samples * delta

    walk(start, end)
    if handler != b"soun":
        return None
    if not timescale:
        raise _Unreadable
    return max(header_duration, sample_duration) / timescale


def _m4a_seconds(data: bytes) -> float:
    tracks = []
    for box_type, body, body_end in _mp4_boxes(data, 0, len(data)):
        if box_type == b"moof":
            # Fragmented files keep sample durations in each fragment; not measured here.
            raise UploadRejected(400, "Fragmented M4A files are not supported.")
        if box_type == b"moov":
            for inner_type, inner_body, inner_end in _mp4_boxes(data, body, body_end):
                if inner_type == b"trak":
                    tracks.append(_mp4_track_seconds(data, inner_body, inner_end))
    audio = [seconds for seconds in tracks if seconds is not None]
    if not audio:
        raise UploadRejected(400, "The M4A file has no audio track.")
    return max(audio)


# --- WebM (EBML / Matroska) ---

# Master elements whose children are scanned; every other element is skipped by size.
_EBML_MASTERS = {
    0x1A45DFA3,  # EBML header
    0x18538067,  # Segment
    0x1549A966,  # Info
    0x1F43B675,  # Cluster
    0xA0,  # BlockGroup
}
_EBML_DOCTYPE, _TIMECODE_SCALE, _DURATION = 0x4282, 0x2AD7B1, 0x4489
_CLUSTER_TIMECODE, _SIMPLE_BLOCK, _BLOCK = 0xE7, 0xA3, 0xA1


def _ebml_vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int, bool]:
    """(value, length in bytes, all value bits set) of the variable-length integer at pos."""
    if pos >= len(data) or data[pos] == 0:
        raise _Unreadable
    length = 9 - data[pos].bit_length()
    if pos + length > len(data):
        raise _Unreadable
    value = int.from_bytes(data[pos:pos + length], "big")
    marker = 1 << (7 * length)
    unknown = value == marker | (marker - 1)
    return (value if keep_marker else value & (marker - 1)), length, unknown


def _webm_seconds(data: bytes) -> float:
    """The longer of the declared Duration and the last block timestamp.

    The file is scanned flat: masters are entered and their children read in
    order, so masters of unknown size (common in browser recordings) work.
    """
    pos, doctype, scale = 0, None, 1_000_000  # TimecodeScale default: 1 ms in ns.
    declared, cluster_time, last_block = 0.0, 0, None
    while pos < len(data):
        element_id, id_length, _ = _ebml_vint(data, pos, keep_marker=True)
        size, size_length, unknown = _ebml_vint(data, pos + id_length, keep_marker=False)
        body = pos + id_length + size_length
        if element_id in _EBML_MASTERS:
            pos = body
            continue
        if unknown or body + size > len(data):
            raise _Unreadable
        value = data[body:body + size]
        if element_id == _EBML_DOCTYPE:
            doctype = value.rstrip(b"\x00")
        elif element_id == _TIMECODE_SCALE:
            scale = int.from_bytes(value, "big")
        elif element_id == _DURATION and size in (4, 8):
            declared = struct.unpack(">f" if size == 4 else ">d", value)[0]
        elif element_id == _CLUSTER_TIMECODE:
            cluster_time = int.from_bytes(value, "big")
        elif element_id in (_SIMPLE_BLOCK, _BLOCK):
            _, track_length, _ = _ebml_vint(value, 0, keep_marker=False)
            if len(value) < track_length + 2:
                raise _Unreadable
            relative = struct.unpack_from(">h", value, track_length)[0]
            last_block = max(last_block or 0, cluster_time + relative)
        pos = body + size
    if doctype != b"webm" or not scale:
        raise UploadRejected(400, "File content does not match its '.webm' extension.")
    if last_block is None and not declared:
        raise _Unreadable
    return max(declared, last_block or 0) * scale / 1e9


# --- Entry point ---

def _signature_matches(data: bytes, extension: str) -> bool:
    if extension == ".mp3":
        return data.startswith(b"ID3") or _mp3_frame(data, 0) is not None
    if extension == ".wav":
        return data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    if extension == ".m4a":
        return data[4:8] == b"ftyp"
    return data.startswith(b"\x1a\x45\xdf\xa3")  # .webm (EBML)


_DURATION_READERS = {".mp3": _mp3_seconds, ".wav": _wav_seconds, ".m4a": _m4a_seconds, ".webm": _webm_seconds}


def audio_seconds(path: Path, extension: str) -> float:
    """Duration of a saved audio file, after checking its real type matches the extension."""
    data = path.read_bytes()  # Already limited to MAX_AUDIO_BYTES.
    if not _signature_matches(data, extension):
        raise UploadRejected(400, f"File content does not match its '{extension}' extension.")
    try:
        return _DURATION_READERS[extension](data)
    except (_Unreadable, struct.error, IndexError, ValueError) as exc:
        raise UploadRejected(400, "Could not read the audio file. It may be corrupted.") from exc


def validate_saved_audio(path: Path, extension: str) -> None:
    seconds = audio_seconds(path, extension)
    if seconds > MAX_AUDIO_SECONDS:
        raise UploadRejected(
            413, f"Audio is {seconds:.0f} seconds long. Maximum is {MAX_AUDIO_SECONDS} seconds."
        )
    if seconds <= 0:
        raise UploadRejected(400, "The audio file is empty.")
