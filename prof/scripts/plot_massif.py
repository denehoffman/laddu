#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "matplotlib>=3.10.7",
#   "pint>=0.25",
# ]
# ///
"""
Convert Valgrind Massif -> PNG.
Usage:
  ./scripts/plot_massif.py massif.out massif.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from pint import UnitRegistry


def parse_massif(path: Path):
    snaps, cur = [], None
    with path.open('r', encoding='utf-8') as f:
        for s in (ln.strip() for ln in f):
            if s.startswith('snapshot='):
                if cur:
                    snaps.append(cur)
                cur = {'snapshot': int(s.split('=')[1])}
            elif s.startswith('time='):
                cur['time'] = int(s.split('=')[1])
            elif s.startswith('mem_heap_B='):
                cur['mem_heap_B'] = int(s.split('=')[1])
            elif s.startswith('mem_heap_extra_B='):
                cur['mem_heap_extra_B'] = int(s.split('=')[1])
            elif s.startswith('mem_stacks_B='):
                cur['mem_stacks_B'] = int(s.split('=')[1])
    if cur:
        snaps.append(cur)
    for d in snaps:
        d.setdefault('mem_heap_B', 0)
        d.setdefault('mem_heap_extra_B', 0)
        d.setdefault('mem_stacks_B', 0)
    return snaps


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    in_path = Path(sys.argv[1])
    png_path = Path(sys.argv[2])

    snaps = parse_massif(in_path)
    if not snaps:
        print('No snapshots found.')
        sys.exit(1)

    rows, peak_idx, peak_total = [], -1, -1
    for s in snaps:
        heap = s['mem_heap_B'] + s['mem_heap_extra_B']
        stack = s['mem_stacks_B']
        total = heap + stack
        if total == 0:
            continue
        rows.append(
            {
                'time_ms': s.get('time', 0),
                'heap_B': heap,
                'stacks_B': stack,
                'total_B': total,
            }
        )
        if total > peak_total:
            peak_total, peak_idx = total, len(rows) - 1

    xs = [r['time_ms'] for r in rows]
    ys = [r['total_B'] for r in rows]
    # ys_heap = [r['heap_B'] for r in rows]
    ys_stack = [r['stacks_B'] for r in rows]
    peak_y = ys[peak_idx]

    ureg = UnitRegistry()
    byte = ureg.parse_units('B')
    reduced_units = (peak_y * byte).to_compact().units
    ys = [(v * byte).to(reduced_units).magnitude for v in ys]
    # ys_heap = [(v * byte).to(reduced_units).magnitude for v in ys_heap]
    ys_stack = [(v * byte).to(reduced_units).magnitude for v in ys_stack]
    peak_y = (peak_y * byte).to(reduced_units).magnitude

    plt.figure()
    plt.title('Memory usage over time (Massif)')
    plt.xlabel('Time (ms)')
    plt.ylabel(f'Total memory usage $({reduced_units:~L})$')
    plt.plot(xs, ys, color='k', label='total')
    plt.fill_between(xs, 0, ys_stack, color='r', label='stack', alpha=0.4)
    plt.fill_between(xs, ys_stack, ys, color='b', label='heap', alpha=0.4)
    plt.axhline(peak_y, ls=':', color='k', label=f'peak: ${peak_y * reduced_units:~L}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=400)


if __name__ == '__main__':
    main()
