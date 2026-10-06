"""ASGI middleware that rejects oversized request bodies before they are received."""

import json

from config import MAX_REQUEST_BYTES, MAX_UPLOAD_BYTES, MB

# Report the file limit users care about, not the internal request margin.
_TOO_LARGE_BODY = json.dumps(
    {"detail": f"File is too large. Maximum size is {MAX_UPLOAD_BYTES // MB} MB."}
).encode()


class RequestSizeLimitMiddleware:
    """Return 413 for requests whose body exceeds MAX_REQUEST_BYTES.

    - With a Content-Length header: rejected immediately, without reading the body.
    - Without one (chunked uploads): the body is counted as it streams in, and
      reading stops as soon as it passes the limit.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await _send_json(send, 400, b'{"detail": "Invalid Content-Length header."}')
                return
            if declared > MAX_REQUEST_BYTES:
                await _send_json(send, 413, _TOO_LARGE_BODY)
                return

        received = 0
        too_large = False
        response_started = False

        async def limited_receive():
            nonlocal received, too_large
            if too_large:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_BYTES:
                    too_large = True
                    # Stop reading; the app sees a disconnect and aborts parsing.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal response_started
            if too_large:
                # Replace whatever error the app produced for the aborted body.
                if not response_started:
                    response_started = True
                    await _send_json(send, 413, _TOO_LARGE_BODY)
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, limited_receive, guarded_send)
        if too_large and not response_started:
            await _send_json(send, 413, _TOO_LARGE_BODY)


async def _send_json(send, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
