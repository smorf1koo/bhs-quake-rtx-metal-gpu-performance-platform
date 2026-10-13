#!/usr/bin/env python3
"""Deterministic 4K test image generator for test task 03.

Creates assets/test_4k.png: 3840x2160, RGB, 8 bits per channel.
The content makes blur errors easy to spot:
  * smooth color gradient in the background (blur must not change brightness);
  * checkerboards with 1, 2, 4, 8, 16, 32 px cells (sharp edges, high frequencies);
  * zone plate (rings of increasing frequency): shows high-frequency attenuation;
  * 1 px lines at the very edges of the frame (border handling, clamp-to-edge);
  * a band of deterministic noise (LCG): the worst case for caches/compression.

Python standard library only. The output is byte-identical on every run.
Generating the 4K image takes about 10 seconds.

    python3 -I tools/gen_test_image.py                 # -> ../assets/test_4k.png
    python3 -I tools/gen_test_image.py --ppm out.ppm   # also write PPM (P6)
    python3 -I tools/gen_test_image.py --width 512 --height 288 --out small.png
"""
import argparse
import math
import os
import struct
import zlib


def png_chunk(tag, data):
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png_rgb(path, width, height, rows):
    """rows: list of bytes (width*3 bytes per row). Sub filter for better compression."""
    comp = zlib.compressobj(9)
    out = []
    for row in rows:
        f = bytearray(len(row) + 1)
        f[0] = 1  # filter type 1 (Sub)
        f[1:4] = row[0:3]
        for i in range(3, len(row)):
            f[i + 1] = (row[i] - row[i - 3]) & 0xFF
        out.append(comp.compress(bytes(f)))
    out.append(comp.flush())
    with open(path, "wb") as fp:
        fp.write(b"\x89PNG\r\n\x1a\n")
        fp.write(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        fp.write(png_chunk(b"IDAT", b"".join(out)))
        fp.write(png_chunk(b"IEND", b""))


def generate_rows(W, H):
    # Scale relative to 4K so that smaller versions look the same.
    s = W / 3840.0
    zp_cx, zp_cy, zp_r = int(2880 * s), int(1080 * s), int(820 * s)
    k_zp = math.pi / (2.0 * zp_r)          # frequency reaches Nyquist at the edge of the circle
    noise_y0, noise_y1 = int(1720 * s), int(1960 * s)
    checker_x0, checker_y0 = int(160 * s), int(240 * s)
    block = max(8, int(256 * s))
    sizes = [1, 2, 4, 8, 16, 32]

    rows = []
    lcg = 0x2545F491
    for y in range(H):
        row = bytearray(W * 3)
        fy = y / (H - 1)
        for x in range(W):
            fx = x / (W - 1)
            r = 40 + 170 * fx
            g = 50 + 150 * fy
            b = 200 - 120 * fx * fy
            # Checkerboards of different scales.
            cx = x - checker_x0
            cy = y - checker_y0
            if 0 <= cy < 2 * block and 0 <= cx < 3 * block:
                idx = (cy // block) * 3 + (cx // block)
                cell = sizes[idx]
                v = 235 if ((cx // cell) + (cy // cell)) % 2 == 0 else 20
                r = g = b = v
            else:
                dx = x - zp_cx
                dy = y - zp_cy
                d2 = dx * dx + dy * dy
                if d2 < zp_r * zp_r:
                    v = 127.5 + 127.5 * math.cos(k_zp * d2)
                    r = g = b = v
            if noise_y0 <= y < noise_y1 and x < W // 2:
                lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
                n = ((lcg >> 8) & 0xFF) - 128
                r += n * 0.5
                g += n * 0.5
                b += n * 0.5
            # 1 px lines along the borders (clamp-to-edge check).
            if x == 0 or y == 0 or x == W - 1 or y == H - 1:
                r, g, b = 255, 255, 255
            elif x == 2 or y == 2 or x == W - 3 or y == H - 3:
                r, g, b = 255, 0, 0
            i = x * 3
            row[i] = min(255, max(0, int(r + 0.5)))
            row[i + 1] = min(255, max(0, int(g + 0.5)))
            row[i + 2] = min(255, max(0, int(b + 0.5)))
        rows.append(bytes(row))
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--out", default=os.path.join(here, "..", "assets", "test_4k.png"))
    ap.add_argument("--ppm", default=None, help="also write PPM (P6)")
    args = ap.parse_args()
    rows = generate_rows(args.width, args.height)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_png_rgb(args.out, args.width, args.height, rows)
    print("written", os.path.normpath(args.out))
    if args.ppm:
        with open(args.ppm, "wb") as f:
            f.write(b"P6\n%d %d\n255\n" % (args.width, args.height))
            for row in rows:
                f.write(row)
        print("written", args.ppm)


if __name__ == "__main__":
    main()
