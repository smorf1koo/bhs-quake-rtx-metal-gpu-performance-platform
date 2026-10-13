#!/usr/bin/env python3
"""Image utilities (Python standard library only).

    python3 -I tools/image_tools.py to-ppm  assets/test_4k.png  test_4k.ppm
    python3 -I tools/image_tools.py to-png  out.ppm             out.png
    python3 -I tools/image_tools.py compare ref.ppm gpu.ppm [--tolerance 1] [--diff diff.png]

PNG: 8-bit RGB and RGBA without interlacing (enough for test_4k.png and most
PNGs saved by image editors). Alpha is dropped when converting to PPM.
PPM: binary P6, maxval 255.

compare prints max/mean abs diff, PSNR and the number of pixels that differ by
more than the tolerance; exit code 0 if max abs diff <= tolerance, otherwise 1.
"""
import argparse
import math
import struct
import sys
import zlib


class Image:
    def __init__(self, width, height, data):
        self.width, self.height, self.data = width, height, data  # data: bytes RGB8

    def __repr__(self):
        return "Image(%dx%d)" % (self.width, self.height)


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("%s: not a PNG" % path)
    pos, idat, hdr = 8, [], None
    while pos < len(blob):
        (length,) = struct.unpack(">I", blob[pos:pos + 4])
        tag = blob[pos + 4:pos + 8]
        data = blob[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", data)
        elif tag == b"IDAT":
            idat.append(data)
        elif tag == b"IEND":
            break
    w, h, depth, ctype, _, _, interlace = hdr
    if depth != 8 or ctype not in (2, 6) or interlace != 0:
        raise ValueError("%s: only 8-bit RGB/RGBA non-interlaced PNG is supported" % path)
    bpp = 3 if ctype == 2 else 4
    raw = zlib.decompress(b"".join(idat))
    stride = w * bpp
    prev = bytearray(stride)
    out = bytearray(w * h * 3)
    p = 0
    for y in range(h):
        ftype = raw[p]
        line = bytearray(raw[p + 1:p + 1 + stride])
        p += 1 + stride
        if ftype == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                upleft = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(left, prev[i], upleft)) & 0xFF
        elif ftype != 0:
            raise ValueError("bad PNG filter %d" % ftype)
        if bpp == 3:
            out[y * w * 3:(y + 1) * w * 3] = line
        else:
            rgb = bytearray(w * 3)
            rgb[0::3], rgb[1::3], rgb[2::3] = line[0::4], line[1::4], line[2::4]
            out[y * w * 3:(y + 1) * w * 3] = rgb
        prev = line
    return Image(w, h, bytes(out))


def write_png(path, img):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    stride = img.width * 3
    raw = bytearray()
    for y in range(img.height):
        raw.append(0)
        raw += img.data[y * stride:(y + 1) * stride]
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", img.width, img.height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        f.write(chunk(b"IEND", b""))


def read_ppm(path):
    with open(path, "rb") as f:
        blob = f.read()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while blob[pos:pos + 1].isspace():
            pos += 1
        if blob[pos:pos + 1] == b"#":
            while blob[pos:pos + 1] not in (b"\n", b""):
                pos += 1
            continue
        start = pos
        while not blob[pos:pos + 1].isspace():
            pos += 1
        tokens.append(blob[start:pos])
    pos += 1  # exactly one whitespace character after maxval
    if tokens[0] != b"P6" or int(tokens[3]) != 255:
        raise ValueError("%s: only binary P6 with maxval 255 is supported" % path)
    w, h = int(tokens[1]), int(tokens[2])
    data = blob[pos:pos + w * h * 3]
    if len(data) != w * h * 3:
        raise ValueError("%s: truncated PPM" % path)
    return Image(w, h, data)


def write_ppm(path, img):
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (img.width, img.height))
        f.write(img.data)


def load(path):
    return read_png(path) if path.lower().endswith(".png") else read_ppm(path)


def save(path, img):
    (write_png if path.lower().endswith(".png") else write_ppm)(path, img)


def compare(a, b, tolerance, diff_path=None):
    if (a.width, a.height) != (b.width, b.height):
        print("size mismatch: %r vs %r" % (a, b))
        return 2
    hist = [0] * 256
    sq = 0
    for x, y in zip(a.data, b.data):
        d = x - y if x >= y else y - x
        hist[d] += 1
        sq += d * d
    n = len(a.data)
    max_diff = max(i for i, c in enumerate(hist) if c)
    mean = sum(i * c for i, c in enumerate(hist)) / n
    mse = sq / n
    psnr = float("inf") if mse == 0 else 10 * math.log10(255 * 255 / mse)
    over = 0
    for i in range(0, n, 3):
        if max(abs(a.data[i + k] - b.data[i + k]) for k in range(3)) > tolerance:
            over += 1
    print("size            %dx%d" % (a.width, a.height))
    print("max_abs_diff    %d" % max_diff)
    print("mean_abs_diff   %.5f" % mean)
    print("psnr_db         %s" % ("inf" if psnr == float("inf") else "%.2f" % psnr))
    print("pixels_over_tol %d (tolerance %d)" % (over, tolerance))
    if diff_path:
        amp = bytes(min(255, abs(x - y) * 32) for x, y in zip(a.data, b.data))
        save(diff_path, Image(a.width, a.height, amp))
        print("diff image      %s (|a-b| x32)" % diff_path)
    return 0 if max_diff <= tolerance else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("to-ppm", "to-png"):
        s = sub.add_parser(name)
        s.add_argument("src")
        s.add_argument("dst")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tolerance", type=int, default=1)
    c.add_argument("--diff", default=None, help="write a difference map (PNG/PPM)")
    args = ap.parse_args()
    if args.cmd == "to-ppm":
        write_ppm(args.dst, load(args.src))
    elif args.cmd == "to-png":
        write_png(args.dst, load(args.src))
    else:
        sys.exit(compare(load(args.a), load(args.b), args.tolerance, args.diff))


if __name__ == "__main__":
    main()
