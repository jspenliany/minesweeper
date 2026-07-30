import time
import logging
import os
from collections import defaultdict

_timings = defaultdict(float)
_counts = defaultdict(int)
_report_path = "timing_report.txt"

def timer(func):
    qualname = getattr(func, '__qualname__', func.__name__)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            _timings[qualname] += elapsed
            _counts[qualname] += 1
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = qualname
    return wrapper

def _format_report():
    lines = []
    total = sum(_timings.values())
    lines.append("")
    lines.append("=" * 70)
    lines.append("TIMING REPORT (cumulative across all games)")
    lines.append("-" * 70)
    lines.append(f"{'Function':<45} {'Count':>6} {'Total(s)':>10} {'Avg(ms)':>10} {'%':>8}")
    lines.append("-" * 70)
    for name in sorted(_timings, key=lambda n: -_timings[n]):
        t = _timings[name]
        c = _counts[name]
        avg = t / c * 1000
        pct = t / total * 100
        lines.append(f"{name:<45} {c:>6} {t:>10.3f} {avg:>10.1f} {pct:>7.1f}%")
    lines.append("-" * 70)
    lines.append(f"{'TOTAL':<45} {sum(_counts.values()):>6} {total:>10.3f} {'':>10} {'':>8}")
    lines.append("=" * 70)
    return lines

def print_report():
    if not _timings:
        logging.info("No timing data collected.")
        return
    lines = _format_report()
    for line in lines:
        print(line)
    try:
        with open(_report_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logging.info(f"Timing report appended to {os.path.abspath(_report_path)}")
    except Exception as e:
        logging.warning(f"Failed to write timing report to file: {e}")

def reset():
    _timings.clear()
    _counts.clear()
