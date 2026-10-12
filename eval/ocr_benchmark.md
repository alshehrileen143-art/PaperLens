# OCR benchmark: PaddleOCR vs RapidOCR

Every page of the vitamin D paper (`samples/1-s2.0-S0021755720302072-main.pdf`, 7 pages), rendered
at 200 DPI and read by OCR. The paper has a text layer, so pypdf's text of each page is the reference.
Local only (no API, no model download). CPU: 8 cores, no GPU.

- **clean:** the page image as rendered.
- **scanned:** the same image tilted 1 to 2 degrees (random sign), Gaussian noise (sigma 8),
  then saved as JPEG quality 30. Seed 1234 + page number, so it can be made again.

Engines (same input images for both):
- **PaddleOCR small:** the project's setup in `ocr.py` (PP-OCRv6 small det/rec, orientation models off,
  oneDNN off).
- **RapidOCR 3.10.0:** onnxruntime 1.30.0, with the PP-OCRv6 small models bundled in the package;
  orientation classification off, no downscaling (`Global.max_side_len: 4000`).

Time is per page, after one warm-up call (model load not counted). Word and number recall count
how many of the reference's words and numbers (with repeats) the OCR text has, in any order, so
reading order does not matter. A misread number shows as missing from the OCR text and, usually,
as an extra one in its place.

Superscript digits and the middle dot are read the same way on both sides before comparing: the text
layer writes "et al.27", "I2" and "128·9" where OCR reads "et al.²⁷", "I²" and "128.9", which
are the same numbers.

## Summary

| Engine | Version | s/page (mean) | s/page (max) | Word recall | Number recall | Numbers missing | Extra numbers |
|---|---|---|---|---|---|---|---|
| PaddleOCR small (current) | clean | 40.1 | 57.0 | 95.8% (3465/3616) | 100.0% (542/542) | 0 | 218 |
| PaddleOCR small (current) | scanned | 36.7 | 57.3 | 95.4% (3451/3616) | 98.9% (536/542) | 6 | 216 |
| RapidOCR (same PP-OCRv6 small models) | clean | 9.9 | 15.8 | 95.8% (3464/3616) | 100.0% (542/542) | 0 | 213 |
| RapidOCR (same PP-OCRv6 small models) | scanned | 10.1 | 13.9 | 95.5% (3455/3616) | 99.4% (539/542) | 3 | 214 |

Tilt of each scanned page: -1.18°, -1.62°, +1.50°, -1.93°, +1.57°, -1.06°, +1.73°.

## Per page, with every number that differs from the reference

### PaddleOCR small (current), clean

| Page | Seconds | Number recall | Missing (in reference, not read) | Extra (read, not in reference) |
|---|---|---|---|---|
| 1 | 31.5 | 40/40 | — | — |
| 2 | 49.1 | 58/58 | — | — |
| 3 | 35.5 | 186/186 | — | — |
| 4 | 36.8 | 12/12 | — | `0.01`, `0.0479`, `0.33`, `0.5`, `0.74`, `0.8`, `0.81`, `0.96`, `0.98` ×3, `1` ×2, `1.01`, `1.03`, `1.04` ×2, `1.05`, `1.08`, `1.10`, `1.12` ×2, `1.13` ×2, `1.14`, `1.17`, `1.18`, `1.20`, `1.22`, `1.23`, `1.26` ×2, `1.29`, `1.30`, `1.32`, `1.38`, `1.39`, `1.40`, `1.41` ×3, `1.43` ×2, `1.48`, `1.50`, `1.52`, `1.53`, `1.55` ×2, `1.59`, `1.65`, `1.74`, `1.76`, `1.86`, `1.91`, `1.92`, `100.0`, `102`, `103`, `104`, `108`, `11`, `112`, `113` ×2, `116`, `119`, `12` ×2, `1234`, `125`, `13`, `131`, `1324`, `135`, `14`, `142`, `155`, `156`, `164`, `175`, `176`, `18`, `18799`, `19`, `1900`, `197`, `198`, `2` ×3, `2.06`, `2.08`, `2.17`, `2.2`, `2.28`, `2.4`, `2.5`, `2.74`, `2.83`, `2.92`, `20` ×2, `2008`, `2010`, `2011` ×4, `2012`, `2013`, `2014` ×3, `2015` ×2, `2016` ×4, `2018` ×2, `2019`, `205`, `21` ×2, `2227`, `2421`, `247`, `2478`, `25`, `2576`, `27`, `28`, `284`, `289`, `295`, `3.18`, `3.44`, `3.53`, `3.7`, `3.78`, `3.8`, `3.92`, `3155`, `32`, `328`, `336`, `348`, `36`, `393`, `4`, `4.1`, `4.45`, `4.7`, `411`, `43`, `44`, `444`, `445`, `459`, `5.0`, `5.5`, `5.7` ×2, `50` ×2, `52`, `54`, `57`, `5801`, `582`, `59` ×2, `6`, `6.2` ×2, `6.5`, `6.8`, `60` ×2, `64`, `65`, `66`, `7.0` ×2, `7.1` ×2, `726`, `73`, `757`, `78`, `79`, `8`, `84`, `852`, `87`, `89`, `9`, `930`, `95`, `961`, `98`, `9814`, `986`, `99` ×2 |
| 5 | 55.0 | 32/32 | — | `0` ×2, `0.3`, `0.5`, `04`, `05`, `1.0`, `2`, `2.0`, `5.0`, `9.6` |
| 6 | 57.0 | 184/184 | — | — |
| 7 | 16.1 | 30/30 | — | — |

### PaddleOCR small (current), scanned

| Page | Seconds | Number recall | Missing (in reference, not read) | Extra (read, not in reference) |
|---|---|---|---|---|
| 1 | 27.3 | 39/40 | `4.0` | `4.01` |
| 2 | 44.7 | 58/58 | — | — |
| 3 | 34.5 | 183/186 | `16`, `19`, `275` | `2` |
| 4 | 35.2 | 11/12 | `22` | `0.01`, `0.0479`, `0.33`, `0.5`, `0.74`, `0.8`, `0.81`, `0.96`, `0.98` ×3, `1` ×2, `1.01`, `1.03`, `1.04` ×2, `1.05`, `1.08`, `1.10`, `1.12` ×2, `1.13` ×2, `1.14`, `1.17`, `1.18`, `1.20`, `1.22`, `1.23`, `1.26` ×2, `1.29`, `1.30`, `1.32`, `1.38`, `1.39`, `1.40`, `1.41` ×3, `1.43` ×2, `1.48`, `1.50`, `1.52`, `1.53`, `1.55` ×2, `1.59`, `1.65`, `1.74`, `1.76`, `1.86`, `1.91`, `1.92`, `100.0`, `102`, `103`, `104`, `108`, `11`, `112`, `113` ×2, `116`, `119`, `12` ×2, `1234`, `125`, `13`, `131`, `1324`, `135`, `14`, `142`, `155`, `156`, `164`, `175`, `176`, `18`, `18799`, `19`, `1900`, `197`, `198`, `2` ×3, `2.06`, `2.08`, `2.17`, `2.2`, `2.28`, `2.4`, `2.5`, `2.74`, `2.83`, `2.92`, `20` ×2, `2008`, `2010`, `2011` ×4, `2012`, `2013`, `2014` ×3, `2015` ×2, `2016` ×4, `2018` ×2, `2019`, `205`, `21` ×2, `222`, `2227`, `2421`, `2478`, `25`, `2576`, `27`, `28`, `284`, `289`, `295`, `3.18`, `3.44`, `3.53`, `3.7`, `3.78`, `3.8`, `3.92`, `3155`, `32`, `328`, `336`, `348`, `36`, `393`, `4`, `4.1`, `4.45`, `4.7`, `411`, `43`, `44`, `444`, `445`, `459`, `5.0`, `5.5`, `5.7` ×2, `50` ×2, `52`, `54`, `57`, `5801`, `582`, `59` ×2, `6`, `6.2` ×2, `6.5`, `6.8`, `60` ×2, `64`, `65`, `66`, `7.0` ×2, `7.1` ×2, `726`, `73`, `757`, `78`, `79`, `8`, `84`, `852`, `87`, `89`, `9`, `930`, `95`, `961`, `9814`, `98247`, `986`, `99` ×2 |
| 5 | 43.1 | 31/32 | `9` | `0`, `0.5`, `00`, `06`, `1.0`, `2.0`, `5.0` |
| 6 | 57.3 | 184/184 | — | — |
| 7 | 14.7 | 30/30 | — | — |

### RapidOCR (same PP-OCRv6 small models), clean

| Page | Seconds | Number recall | Missing (in reference, not read) | Extra (read, not in reference) |
|---|---|---|---|---|
| 1 | 7.8 | 40/40 | — | — |
| 2 | 12.6 | 58/58 | — | — |
| 3 | 8.6 | 186/186 | — | — |
| 4 | 9.4 | 12/12 | — | `0.01`, `0.0479`, `0.33`, `0.5`, `0.74`, `0.8`, `0.81`, `0.96`, `0.98` ×3, `1`, `1.01`, `1.03`, `1.04` ×2, `1.05`, `1.08`, `1.10`, `1.12` ×2, `1.13` ×2, `1.14`, `1.17`, `1.18`, `1.20`, `1.22`, `1.23`, `1.26` ×2, `1.29`, `1.30`, `1.32`, `1.38`, `1.39`, `1.40`, `1.41` ×3, `1.43` ×2, `1.48`, `1.50`, `1.52`, `1.53`, `1.55` ×2, `1.59`, `1.65`, `1.74`, `1.76`, `1.86`, `1.91`, `1.92`, `100.0`, `102`, `103`, `104`, `108`, `11`, `112`, `113` ×2, `116`, `119`, `12` ×2, `1234`, `125`, `13`, `131`, `1324`, `135`, `14`, `142`, `155`, `156`, `164`, `175`, `176`, `18`, `18799`, `19`, `1900`, `197`, `198`, `2` ×3, `2.06`, `2.08`, `2.17`, `2.2`, `2.28`, `2.4`, `2.5`, `2.74`, `2.83`, `2.92`, `20` ×2, `2008`, `2010`, `2011` ×4, `2012`, `2013`, `2014` ×3, `2015` ×2, `2016` ×4, `2018` ×2, `2019`, `205`, `21` ×2, `2227`, `2421`, `247`, `2478`, `25`, `2576`, `27`, `28`, `284`, `289`, `295`, `3.18`, `3.44`, `3.53`, `3.7`, `3.78`, `3.8`, `3.92`, `3155`, `32`, `328`, `336`, `348`, `36`, `393`, `4`, `4.1`, `4.45`, `4.7`, `411`, `43`, `44`, `444`, `445`, `459`, `5.0`, `5.5`, `5.7` ×2, `50` ×2, `52`, `54`, `57`, `5801`, `582`, `59` ×2, `6`, `6.2` ×2, `6.5`, `6.8`, `60` ×2, `64`, `65`, `66`, `7.0` ×2, `7.1` ×2, `726`, `73`, `757`, `78`, `79`, `8`, `84`, `852`, `87`, `89`, `9`, `930`, `95`, `961`, `98`, `9814`, `986`, `99` ×2 |
| 5 | 10.9 | 32/32 | — | `0` ×2, `0.5`, `00`, `1.0`, `2.0`, `5.0` |
| 6 | 15.8 | 184/184 | — | — |
| 7 | 4.2 | 30/30 | — | — |

### RapidOCR (same PP-OCRv6 small models), scanned

| Page | Seconds | Number recall | Missing (in reference, not read) | Extra (read, not in reference) |
|---|---|---|---|---|
| 1 | 10.2 | 39/40 | `4.0` | `4.01` |
| 2 | 11.8 | 58/58 | — | — |
| 3 | 9.2 | 184/186 | `16`, `275` | `2` |
| 4 | 10.1 | 12/12 | — | `0.01`, `0.0479`, `0.33`, `0.5`, `0.74`, `0.8`, `0.81`, `0.96`, `0.98` ×3, `1` ×2, `1.01`, `1.03`, `1.04` ×2, `1.05`, `1.08`, `1.10`, `1.12` ×2, `1.13` ×2, `1.14`, `1.17`, `1.18`, `1.20`, `1.22`, `1.23`, `1.26` ×2, `1.29`, `1.30`, `1.32`, `1.38`, `1.39`, `1.40`, `1.41` ×3, `1.43` ×2, `1.48`, `1.50`, `1.52`, `1.53`, `1.55` ×2, `1.59`, `1.65`, `1.74`, `1.76`, `1.86`, `1.91`, `1.92`, `100.0`, `102`, `103`, `104`, `108`, `11`, `112`, `113` ×2, `116`, `119`, `12` ×2, `1234`, `125`, `13`, `131`, `1324`, `135`, `14`, `142`, `155`, `156`, `164`, `175`, `176`, `18`, `18799`, `19`, `1900`, `197`, `198`, `2` ×3, `2.06`, `2.08`, `2.17`, `2.2`, `2.28`, `2.4`, `2.5`, `2.74`, `2.83`, `2.92`, `20` ×2, `2008`, `2010`, `2011` ×4, `2012`, `2013`, `2014` ×3, `2015` ×2, `2016` ×4, `2018` ×2, `2019`, `205`, `21` ×2, `2227`, `2421`, `247`, `2478`, `25`, `2576`, `27`, `28`, `284`, `289`, `295`, `3.18`, `3.44`, `3.53`, `3.7`, `3.78`, `3.8`, `3.92`, `3155`, `32`, `328`, `336`, `348`, `36`, `393`, `4`, `4.1`, `4.45`, `4.7`, `411`, `43`, `44`, `444`, `445`, `459`, `5.0`, `5.5`, `5.7` ×2, `50` ×2, `52`, `54`, `57`, `5801`, `582`, `59` ×2, `6`, `6.2` ×2, `6.5`, `6.8`, `60` ×2, `64`, `65`, `66`, `7.0` ×2, `7.1` ×2, `726`, `73`, `757`, `78`, `79`, `8`, `84`, `852`, `87`, `89`, `9`, `930`, `95`, `961`, `98`, `9814`, `986`, `99` ×2 |
| 5 | 11.0 | 32/32 | — | `0.5`, `1.0`, `2.0`, `22`, `5.0` |
| 6 | 13.9 | 184/184 | — | — |
| 7 | 4.3 | 30/30 | — | — |

## Reading the results

- **"Extra numbers" are not errors on pages 4 and 5.** Page 4 holds two images (the search flowchart and
  the forest plot) and page 5 one (a plot). Their numbers are not in the text layer, so the reference
  lacks them; OCR reads them. Both engines read nearly the same list there.
- **Clean pages:** both engines read every number of the reference (542/542). Word recall is the same
  (95.8%); the missing words are mostly the text layer's ligatures and hyphenation, not misreadings.
- **Simulated scans** (tilt, noise, JPEG 30):
  - PaddleOCR: 6 numbers not read: `4.0` read as `4.01` (page 1), `16`, `19` and `275` (page 3), `22`
    (page 4) and `9` (page 5).
  - RapidOCR: 3 numbers not read: `4.0` read as `4.01` (page 1), `16` and `275` (page 3). Each of these
    was also missed by PaddleOCR.
- **Time per page:** PaddleOCR 40.1 s on average (57.3 s at most); RapidOCR 9.9 to 10.1 s (15.8 s at
  most): about 4 times faster, with the same models.
- **Safe OCR page limit** for the app's 300 s timeout, with 80 s kept for the other steps (model load,
  checks, embeddings or the summary call at their worst), from the slowest page:
  - PaddleOCR: (300 − 80) ÷ 57.3 = 3.8, so 3 pages (today's limit of 10 needs about 400 s).
  - RapidOCR: (300 − 80) ÷ 15.8 = 13.9, so 13 pages (about 130 s at the average speed).

Limits of this test: one paper (7 pages, two columns, a born-digital PDF), simulated scan damage only
(no real scanner, no handwriting, no colour background), English only, and one run per page.
The script is `eval/ocr_benchmark.py` (its input images are made in a temporary folder).
