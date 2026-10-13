#!/usr/bin/env python3
"""Charts from results.csv and thermal.csv (test task 03).

    python3 -I tools/plot_results.py results.csv                  # -> results.svg (or .png)
    python3 -I tools/plot_results.py results.csv -o results.png   # PNG needs matplotlib
    python3 -I tools/plot_results.py --thermal thermal.csv        # -> thermal.svg

Without matplotlib the script draws SVG with the standard library only.
With matplotlib (pip install matplotlib) it can save PNG/PDF/SVG.

results.csv: horizontal bars of median_ms per variant, whiskers = min..max.
If the file has several devices or sigma values, each combination gets its
own panel (one shared X axis per panel, no dual scales).
thermal.csv: a gpu_ms vs elapsed_s line for each (device, variant).
Lines starting with '#' are ignored.
"""
import argparse
import csv
import math
import os
import sys
from collections import OrderedDict
from xml.sax.saxutils import escape

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order, at most 3 lines per panel
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def fnum(row, key):
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return float("nan")


def nice_max(v):
    """Nearest "round" axis limit above v (1, 1.2, 1.5, 2, ... x 10^k)."""
    if not v > 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if step * mag >= v:
            return step * mag
    return 10 * mag


def ticks(top, n=5):
    return [top * i / n for i in range(n + 1)]


# ----------------------------------------------------------------- results.csv
def group_results(rows):
    panels = OrderedDict()
    for r in rows:
        key = (r.get("device", "?"), r.get("sigma", "?"), r.get("width", "?"), r.get("height", "?"))
        panels.setdefault(key, []).append(r)
    return panels


def svg_results(rows, out):
    panels = group_results(rows)
    W, bar_h, gap, left, right = 860, 22, 12, 170, 90
    parts, y = [], 20
    for (device, sigma, w, h), items in panels.items():
        items = sorted(items, key=lambda r: fnum(r, "median_ms"))
        vmax = nice_max(max(max(fnum(r, "max_ms"), fnum(r, "median_ms")) for r in items))
        plot_w = W - left - right
        sx = lambda v: left + plot_w * v / vmax
        parts.append('<text x="%d" y="%d" font-size="15" font-weight="600" fill="%s">%s</text>'
                     % (left - 150, y + 14, TEXT, escape("%s · %sx%s · sigma %s" % (device, w, h, sigma))))
        parts.append('<text x="%d" y="%d" font-size="12" fill="%s">GPU time, ms (median; whiskers = min..max). Lower is better.</text>'
                     % (left - 150, y + 32, TEXT_2))
        top = y + 48
        height = len(items) * (bar_h + gap)
        for t in ticks(vmax):
            x = sx(t)
            parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1"/>' % (x, top - 4, x, top + height, GRID))
            parts.append('<text x="%.1f" y="%d" font-size="11" fill="%s" text-anchor="middle">%g</text>' % (x, top + height + 16, TEXT_2, round(t, 3)))
        for i, r in enumerate(items):
            by = top + i * (bar_h + gap)
            med, lo, hi = fnum(r, "median_ms"), fnum(r, "min_ms"), fnum(r, "max_ms")
            tip = "%s: median %.3f ms, min %.3f, max %.3f, runs %s, max_abs_diff %s" % (
                r.get("variant"), med, lo, hi, r.get("measured_runs"), r.get("max_abs_diff"))
            parts.append('<g><title>%s</title>' % escape(tip))
            parts.append('<text x="%d" y="%d" font-size="13" fill="%s" text-anchor="end">%s</text>'
                         % (left - 10, by + bar_h - 6, TEXT, escape(r.get("variant", "?"))))
            bw = max(1.0, sx(med) - left)
            parts.append('<path d="M%d,%d h%.1f a4,4 0 0 1 4,4 v%d a4,4 0 0 1 -4,4 h-%.1f z" fill="%s"/>'
                         % (left, by, bw - 4, bar_h - 8, bw - 4, SERIES[0]))
            if lo == lo and hi == hi:
                cy = by + bar_h / 2
                parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.5"/>' % (sx(lo), cy, sx(hi), cy, TEXT_2))
                for xv in (lo, hi):
                    parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.5"/>' % (sx(xv), cy - 5, sx(xv), cy + 5, TEXT_2))
            parts.append('<text x="%.1f" y="%d" font-size="12" fill="%s">%.3f ms</text>'
                         % (max(sx(hi) if hi == hi else 0, sx(med)) + 8, by + bar_h - 6, TEXT, med))
            parts.append('</g>')
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>' % (left, top - 4, left, top + height, TEXT_2))
        y = top + height + 44
    write_svg(out, W, y, parts)


def mpl_results(rows, out, plt):
    panels = group_results(rows)
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 1.2 + 0.5 * max(len(v) for v in panels.values()) * len(panels) + 0.8 * len(panels)), squeeze=False)
    for ax, ((device, sigma, w, h), items) in zip(axes[:, 0], panels.items()):
        items = sorted(items, key=lambda r: fnum(r, "median_ms"), reverse=True)
        names = [r.get("variant", "?") for r in items]
        med = [fnum(r, "median_ms") for r in items]
        lo = [m - fnum(r, "min_ms") for m, r in zip(med, items)]
        hi = [fnum(r, "max_ms") - m for m, r in zip(med, items)]
        ax.barh(names, med, color=SERIES[0], height=0.6, xerr=[lo, hi], ecolor=TEXT_2, capsize=4)
        for i, m in enumerate(med):
            ax.text(m + hi[i] if hi[i] == hi[i] else m, i, "  %.3f ms" % m, va="center", fontsize=9, color=TEXT)
        ax.set_title("%s · %sx%s · sigma %s" % (device, w, h, sigma), loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel("GPU time, ms (median; whiskers = min..max). Lower is better.", color=TEXT_2, fontsize=9)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)


# ----------------------------------------------------------------- thermal.csv
def group_thermal(rows):
    series = OrderedDict()
    for r in rows:
        series.setdefault("%s · %s" % (r.get("device", "?"), r.get("variant", "?")), []).append(
            (fnum(r, "elapsed_s"), fnum(r, "gpu_ms")))
    if len(series) > len(SERIES):
        sys.exit("thermal.csv: more than %d series, split the file (up to 3 lines per panel)" % len(SERIES))
    return series


def svg_thermal(rows, out):
    series = group_thermal(rows)
    W, H, left, right, top, bottom = 860, 420, 70, 30, 70, 50
    xs = [p[0] for s in series.values() for p in s]
    ys = [p[1] for s in series.values() for p in s]
    xmax, ymax = nice_max(max(xs)), nice_max(max(ys))
    pw, ph = W - left - right, H - top - bottom
    sx = lambda v: left + pw * v / xmax
    sy = lambda v: top + ph * (1 - v / ymax)
    parts = ['<text x="%d" y="28" font-size="15" font-weight="600" fill="%s">GPU time under sustained load</text>' % (left, TEXT),
             '<text x="%d" y="48" font-size="12" fill="%s">ms per iteration vs. elapsed time, s. Rising line = throttling.</text>' % (left, TEXT_2)]
    for t in ticks(ymax):
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>' % (left, sy(t), W - right, sy(t), GRID))
        parts.append('<text x="%d" y="%.1f" font-size="11" fill="%s" text-anchor="end">%g</text>' % (left - 8, sy(t) + 4, TEXT_2, round(t, 3)))
    for t in ticks(xmax):
        parts.append('<text x="%.1f" y="%d" font-size="11" fill="%s" text-anchor="middle">%g</text>' % (sx(t), H - bottom + 18, TEXT_2, round(t, 1)))
    for i, (name, pts) in enumerate(series.items()):
        pts.sort()
        d = " ".join(("M" if j == 0 else "L") + "%.1f,%.1f" % (sx(x), sy(y)) for j, (x, y) in enumerate(pts))
        parts.append('<path d="%s" fill="none" stroke="%s" stroke-width="2"><title>%s</title></path>' % (d, SERIES[i], escape(name)))
        if len(series) > 1:
            lx, ly = pts[-1]
            parts.append('<text x="%.1f" y="%.1f" font-size="12" fill="%s" text-anchor="end">%s</text>' % (sx(lx), sy(ly) - 8, TEXT, escape(name)))
            parts.append('<rect x="%d" y="%d" width="12" height="3" fill="%s"/><text x="%d" y="%d" font-size="12" fill="%s">%s</text>'
                         % (left + 220 * i, H - 14, SERIES[i], left + 220 * i + 18, H - 9, TEXT, escape(name)))
    write_svg(out, W, H, parts)


def mpl_thermal(rows, out, plt):
    series = group_thermal(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, pts) in enumerate(series.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[i], linewidth=2, label=name)
    ax.set_title("GPU time under sustained load", loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel("elapsed, s", color=TEXT_2)
    ax.set_ylabel("GPU time per iteration, ms", color=TEXT_2)
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def write_svg(path, w, h, parts):
    with open(path, "w", encoding="utf-8") as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="%s">\n'
                % (w, h, w, h, escape(FONT, {"'": "&apos;"})))
        f.write('<rect width="100%%" height="100%%" fill="%s"/>\n' % SURFACE)
        f.write("\n".join(parts))
        f.write("\n</svg>\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", help="results.csv")
    ap.add_argument("--thermal", help="thermal.csv")
    ap.add_argument("-o", "--out", help="output file (.svg; .png/.pdf need matplotlib)")
    ap.add_argument("--no-matplotlib", action="store_true", help="always use the built-in SVG writer")
    args = ap.parse_args()
    if not args.results and not args.thermal:
        ap.error("pass results.csv and/or --thermal thermal.csv")

    plt = None
    if not args.no_matplotlib:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F811
        except ImportError:
            plt = None

    jobs = []
    if args.results:
        jobs.append(("results", args.results, svg_results, mpl_results))
    if args.thermal:
        jobs.append(("thermal", args.thermal, svg_thermal, mpl_thermal))
    for name, src, svg_fn, mpl_fn in jobs:
        rows = read_csv(src)
        if not rows:
            sys.exit("%s: no data rows (header only?)" % src)
        out = args.out if (args.out and len(jobs) == 1) else os.path.splitext(src)[0] + (
            os.path.splitext(args.out)[1] if args.out else ".svg")
        ext = os.path.splitext(out)[1].lower()
        if plt is not None:
            mpl_fn(rows, out, plt)
        elif ext == ".svg":
            svg_fn(rows, out)
        else:
            sys.exit("matplotlib is not installed: only .svg output is available")
        print("written", out)


if __name__ == "__main__":
    main()
