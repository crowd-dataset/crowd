"""Check parse_gps on overlay texts tesseract read from dashcam videos: each format the pipeline knows, the
longitude-first one, and numbers from two OCR lines that are no position (2026-10-23: Maracaibo's dashcam).
Run from the repo with the crowd env's python."""
import propose_segments as ps

cases = {
    'W71.6482,N10.6686': (10.6686, -71.6482),
    '2024-12-05 10:01:25 47/ W71.6482,N10.6686': (10.6686, -71.6482),
    'N40.12345 W75.12345': (40.12345, -75.12345),
    '40.12345N 75.12345W': (40.12345, -75.12345),
    '40.123456, -75.123456': (40.123456, -75.123456),
    'GARMIN 29-05-2021 12:05:10 49.29065 -123.13081': (49.29065, -123.13081),
    'lat: 23.18409 lon: 120.48879': (23.18409, 120.48879),
    # a colon after the hemisphere, OCR's ",." and a space after the point (kpGeawJLR3o, 2026-11-17)
    'N:51,.8796 W:2.2013 7 - - 16-10-2026 08:40:06': (51.8796, -2.2013),
    'N:51.8798 W:2. 1963': (51.8798, -2.1963),
    'N: 3 of 12, E: 2 x': None,
    '10.6686\n\x0c\n1.6482': None,  # the end of one line and a cut-off number on the next
    '': None,
}
for text, want in cases.items():
    got = ps.parse_gps(text)
    assert got == want or (got and want and all(abs(a - b) < 1e-9 for a, b in zip(got, want))), (text, got, want)
print('ok')
