"""
WiFi CSI Heatmap & Analysis — Post-processing collected zone data

Takes the CSV from collect_zones.py and generates:
  1. Heatmap of WiFi motion intensity per zone
  2. Zone classification accuracy (if labeled)
  3. Time-series plots
  4. Summary statistics for your professor

Usage:
  python visualize.py data/zones_20260720_143000.csv
  python visualize.py data/  # processes all CSVs in the folder

Output: PNG heatmaps + analysis summary in the data/ folder
"""

import csv
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend (works without display)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed. Run: pip install matplotlib")
    print("       Will generate text summary only.")


def load_csv(path):
    """Load zone collection CSV into structured data."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'time': float(row['timestamp']),
                'elapsed': float(row['elapsed_s']),
                'zone_col': int(row['zone_col']),
                'zone_row': int(row['zone_row']),
                'zone': row['zone_label'],
                'esp1': float(row['esp1_movement']),
                'esp2': float(row['esp2_movement']),
                'esp1_motion': int(row['esp1_motion']),
                'esp2_motion': int(row['esp2_motion']),
                'px': int(row['person_x']) if row['person_x'] else 0,
                'py': int(row['person_y']) if row['person_y'] else 0,
                'px_norm': float(row['person_x_norm']) if row['person_x_norm'] else 0,
                'py_norm': float(row['person_y_norm']) if row['person_y_norm'] else 0,
            })
    return rows


def analyze(rows):
    """Compute per-zone statistics."""
    zones = defaultdict(lambda: {'esp1': [], 'esp2': [], 'count': 0})
    max_col, max_row = 0, 0

    for r in rows:
        key = (r['zone_col'], r['zone_row'])
        zones[key]['esp1'].append(r['esp1'])
        zones[key]['esp2'].append(r['esp2'])
        zones[key]['count'] += 1
        max_col = max(max_col, r['zone_col'])
        max_row = max(max_row, r['zone_row'])

    cols = max_col + 1
    rws = max_row + 1

    # Build matrices
    esp1_avg = np.zeros((rws, cols))
    esp2_avg = np.zeros((rws, cols))
    combined_avg = np.zeros((rws, cols))
    sample_counts = np.zeros((rws, cols), dtype=int)

    for (c, r), data in zones.items():
        esp1_avg[r, c] = np.mean(data['esp1'])
        esp2_avg[r, c] = np.mean(data['esp2'])
        combined_avg[r, c] = np.mean([max(a, b) for a, b in zip(data['esp1'], data['esp2'])])
        sample_counts[r, c] = data['count']

    return {
        'cols': cols, 'rows': rws,
        'esp1_avg': esp1_avg, 'esp2_avg': esp2_avg,
        'combined_avg': combined_avg, 'counts': sample_counts,
        'zones': zones, 'total_samples': len(rows),
        'duration': rows[-1]['elapsed'] if rows else 0,
    }


def plot_heatmaps(stats, output_path):
    """Generate publication-quality heatmap figure."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('WiFi CSI Motion Heatmap — Zone Localization Experiment', fontsize=14, fontweight='bold')

    titles = ['ESP32-1 Avg Movement', 'ESP32-2 Avg Movement', 'Combined (Max) Movement']
    data = [stats['esp1_avg'], stats['esp2_avg'], stats['combined_avg']]

    for ax, title, d in zip(axes, titles, data):
        im = ax.imshow(d, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(d.max(), 1))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        # Add value annotations
        for r in range(stats['rows']):
            for c in range(stats['cols']):
                val = d[r, c]
                color = 'white' if val > d.max() * 0.5 else 'black'
                ax.text(c, r, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

        # Zone labels
        ax.set_xticks(range(stats['cols']))
        ax.set_yticks(range(stats['rows']))
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Heatmap → {output_path}")


def plot_timeseries(rows, output_path):
    """Plot movement scores over time."""
    if not HAS_MPL:
        return

    times = [r['elapsed'] for r in rows]
    esp1 = [r['esp1'] for r in rows]
    esp2 = [r['esp2'] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle('WiFi CSI Movement Score — Time Series', fontsize=13, fontweight='bold')

    ax1.plot(times, esp1, 'b-', alpha=0.7, linewidth=0.8, label='ESP32-1')
    ax1.plot(times, esp2, 'r-', alpha=0.7, linewidth=0.8, label='ESP32-2')
    ax1.set_ylabel('Movement Score (0-10)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 10)

    # Zone over time
    zone_labels = [f"({r['zone_col']},{r['zone_row']})" for r in rows]
    unique_zones = sorted(set(zone_labels))
    zone_nums = [unique_zones.index(z) for z in zone_labels]
    ax2.scatter(times, zone_nums, c='green', s=2, alpha=0.5)
    ax2.set_ylabel('Zone')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_yticks(range(len(unique_zones)))
    ax2.set_yticklabels(unique_zones, fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Time series → {output_path}")


def plot_zone_fingerprints(stats, output_path):
    """Bar chart showing WiFi 'fingerprint' for each zone — shows separability."""
    if not HAS_MPL:
        return

    zones = []
    e1_vals = []
    e2_vals = []

    for r in range(stats['rows']):
        for c in range(stats['cols']):
            if stats['counts'][r, c] > 0:
                zones.append(f"Z({c},{r})")
                e1_vals.append(stats['esp1_avg'][r, c])
                e2_vals.append(stats['esp2_avg'][r, c])

    x = np.arange(len(zones))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, e1_vals, width, label='ESP32-1', color='steelblue')
    ax.bar(x + width/2, e2_vals, width, label='ESP32-2', color='indianred')
    ax.set_xlabel('Zone')
    ax.set_ylabel('Average Movement Score')
    ax.set_title('WiFi CSI Fingerprint per Zone — ESP32-1 vs ESP32-2', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Fingerprints → {output_path}")


def print_summary(stats):
    """Print text summary."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Grid:           {stats['cols']} x {stats['rows']} = {stats['cols'] * stats['rows']} zones")
    print(f"  Total samples:  {stats['total_samples']}")
    print(f"  Duration:       {stats['duration']:.0f} seconds")
    print()
    print("  Per-zone average movement scores:")
    print(f"  {'Zone':<10} {'ESP1':>8} {'ESP2':>8} {'Combined':>10} {'Samples':>8}")
    print("  " + "-" * 48)
    for r in range(stats['rows']):
        for c in range(stats['cols']):
            if stats['counts'][r, c] > 0:
                print(f"  Z({c},{r}){'':<5} {stats['esp1_avg'][r,c]:>8.3f} "
                      f"{stats['esp2_avg'][r,c]:>8.3f} "
                      f"{stats['combined_avg'][r,c]:>10.3f} "
                      f"{stats['counts'][r,c]:>8}")
    print()

    # Separability check
    vals = stats['combined_avg'][stats['counts'] > 0]
    if len(vals) > 1 and vals.max() > 0:
        spread = vals.std() / vals.mean() if vals.mean() > 0 else 0
        print(f"  Signal spread (CoV): {spread:.2f}")
        if spread > 0.3:
            print("  ✓ Good separability — zones have distinct WiFi signatures!")
        elif spread > 0.1:
            print("  ~ Moderate separability — may need more data or better placement")
        else:
            print("  ✗ Low separability — try moving ESP32s further from router")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <csv_file_or_folder>")
        print("       python visualize.py data/zones_20260720_143000.csv")
        return

    target = Path(sys.argv[1])

    if target.is_dir():
        csvs = sorted(target.glob("**/zones_*.csv"))
        if not csvs:
            print(f"No zones_*.csv files found in {target}")
            return
        csv_path = csvs[-1]  # latest
        print(f"[INFO] Using latest: {csv_path}")
    else:
        csv_path = target

    # Load data
    rows = load_csv(csv_path)
    if not rows:
        print("No data in CSV"); return
    print(f"[INFO] Loaded {len(rows)} samples from {csv_path}")

    # Analyze
    stats = analyze(rows)
    print_summary(stats)

    # Generate plots
    out_dir = csv_path.parent
    stem = csv_path.stem

    if HAS_MPL:
        plot_heatmaps(stats, out_dir / f"{stem}_heatmap.png")
        plot_timeseries(rows, out_dir / f"{stem}_timeseries.png")
        plot_zone_fingerprints(stats, out_dir / f"{stem}_fingerprints.png")
        print(f"\n[DONE] All plots saved in {out_dir}/")
    else:
        print("\n[INFO] Install matplotlib for visual plots: pip install matplotlib")


if __name__ == '__main__':
    main()
