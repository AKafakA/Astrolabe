#!/usr/bin/env python3
"""
Extra Ablation: Plot Po2 (N=2) comparison with Astrolabe (N=12) and Llumnix-.

Shows that power-of-two-choices with N=2 achieves comparable performance
while reducing prediction overhead by 6x.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import re

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "experiment_output/data")
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_output/results/extra_ablation")

# Default system name (can be overridden via --system-name argument)
SYSTEM_NAME = "Astrolabe"

# Color scheme
COLORS = {
    'po2': '#e74c3c',      # Red for Po2
    'astrolabe': '#2ecc71',    # Green for Astrolabe
    'llumnix': '#3498db',  # Blue for Llumnix-
}

MARKERS = {
    'po2': 'o',
    'astrolabe': 's',
    'llumnix': '^',
}

def get_labels(system_name="Astrolabe"):
    return {
        'po2': f'{system_name}-Po2 (N=2)',
        'astrolabe': f'{system_name} (N=12)',
        'llumnix': 'Llumnix-',
    }

# Labels (will be updated in main based on --system-name)
LABELS = get_labels()


def find_npz(base_dir, qps, pattern_suffix=None):
    """Find npz file for a given QPS, optionally matching a pattern suffix.

    Returns the LAST matching directory (sorted order) to match experiment_plot.py behavior.
    """
    if not os.path.exists(base_dir):
        return None
    result = None
    for d in sorted(os.listdir(base_dir)):
        match = re.match(r'qps_(\d+\.?\d*)_', d)
        if match and float(match.group(1)) == float(qps):
            if pattern_suffix and pattern_suffix not in d:
                continue
            npz_path = os.path.join(base_dir, d, "benchmark_all_metrics.npz")
            if os.path.exists(npz_path):
                result = npz_path  # Keep last match
    return result


def get_metrics(npz_path):
    """Load all metrics from npz file."""
    if npz_path is None:
        return None
    try:
        data = np.load(npz_path)
        latencies = data['request_latencies']
        ttft = data['prefill_token_latencies'] if 'prefill_token_latencies' in data else None
        return {
            'mean_latency': np.mean(latencies),
            'p99_latency': np.percentile(latencies, 99),
            'p99_ttft': np.percentile(ttft, 99) if ttft is not None else 0,
            'throughput': float(data['Throughput']),
            'scheduling_overhead': np.mean(data['scheduling_overhead']) if 'scheduling_overhead' in data else 0,
        }
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None


def get_all_qps_values(base_dir, pattern_suffix=None):
    """Get all available QPS values from a directory."""
    qps_values = set()
    if not os.path.exists(base_dir):
        return qps_values
    for d in os.listdir(base_dir):
        match = re.match(r'qps_(\d+\.?\d*)_', d)
        if match:
            if pattern_suffix and pattern_suffix not in d:
                continue
            qps_values.add(float(match.group(1)))
    return qps_values


def load_results():
    """Load results from all three systems."""
    po2_dir = os.path.join(DATA_DIR, "extra_ablation/po2/sharegpt/min_new_request_latency")
    block_dir = os.path.join(DATA_DIR, "main/sharegpt/min_new_request_latency")
    llumnix_dir = os.path.join(DATA_DIR, "main/sharegpt/min_lunmnix_load")

    block_pattern = "len_estimated_true"
    llumnix_pattern = "len_estimated_false"

    po2_qps = get_all_qps_values(po2_dir)
    block_qps = get_all_qps_values(block_dir, block_pattern)
    llumnix_qps = get_all_qps_values(llumnix_dir, llumnix_pattern)

    common_qps = sorted(po2_qps & block_qps & llumnix_qps)
    print(f"Found {len(common_qps)} common QPS values")

    results = {
        'qps': [],
        'po2_mean': [], 'po2_p99': [], 'po2_ttft_p99': [], 'po2_overhead': [],
        'block_mean': [], 'block_p99': [], 'block_ttft_p99': [], 'block_overhead': [],
        'llumnix_mean': [], 'llumnix_p99': [], 'llumnix_ttft_p99': [], 'llumnix_overhead': [],
    }

    for qps in common_qps:
        po2_metrics = get_metrics(find_npz(po2_dir, qps))
        block_metrics = get_metrics(find_npz(block_dir, qps, block_pattern))
        llumnix_metrics = get_metrics(find_npz(llumnix_dir, qps, llumnix_pattern))

        if po2_metrics and block_metrics and llumnix_metrics:
            results['qps'].append(qps)
            for prefix, metrics in [('po2', po2_metrics), ('astrolabe', block_metrics), ('llumnix', llumnix_metrics)]:
                results[f'{prefix}_mean'].append(metrics['mean_latency'] / 1000)
                results[f'{prefix}_p99'].append(metrics['p99_latency'] / 1000)
                results[f'{prefix}_ttft_p99'].append(metrics['p99_ttft'] / 1000)
                results[f'{prefix}_overhead'].append(metrics['scheduling_overhead'])

    return results


def calculate_capacity(qps_list, p99_list, slo):
    """Calculate capacity: max QPS where P99 latency is below SLO."""
    qps_arr = np.array(qps_list)
    p99_arr = np.array(p99_list)
    above_slo = p99_arr > slo
    if not any(above_slo):
        return float(qps_arr[-1])
    if all(above_slo):
        return 0.0
    first_above_idx = np.where(above_slo)[0][0]
    if first_above_idx == 0:
        return 0.0
    return float(qps_arr[first_above_idx - 1])


def load_system_capacity_data(base_dir, pattern_suffix=None):
    """Load all QPS and TTFT P99 data for capacity calculation."""
    qps_list = []
    ttft_list = []
    all_qps = get_all_qps_values(base_dir, pattern_suffix)
    for qps in sorted(all_qps):
        npz_path = find_npz(base_dir, qps, pattern_suffix)
        if npz_path:
            metrics = get_metrics(npz_path)
            if metrics and metrics['p99_ttft'] > 0:
                qps_list.append(qps)
                ttft_list.append(metrics['p99_ttft'] / 1000)
    return qps_list, ttft_list


def plot_single_metric(ax, qps, po2_data, block_data, llumnix_data, ylabel, title, show_legend=True):
    """Helper to plot a single metric with consistent styling."""
    ax.plot(qps, po2_data, f'{MARKERS["po2"]}-', color=COLORS['po2'], linewidth=2,
            markersize=8, label=LABELS['po2'], markeredgecolor='black')
    ax.plot(qps, block_data, f'{MARKERS["astrolabe"]}-', color=COLORS['astrolabe'], linewidth=2,
            markersize=8, label=LABELS['astrolabe'], markeredgecolor='black')
    ax.plot(qps, llumnix_data, f'{MARKERS["llumnix"]}--', color=COLORS['llumnix'], linewidth=2,
            markersize=8, label=LABELS['llumnix'], markeredgecolor='black')

    ax.set_xlabel('QPS', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    if show_legend:
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(qps[::2])
    ax.tick_params(axis='x', rotation=45)


def plot_capacity_bar(ax, ttft_slo=3.0):
    """Plot capacity bar chart showing max QPS under TTFT P99 SLO constraint."""
    po2_dir = os.path.join(DATA_DIR, "extra_ablation/po2/sharegpt/min_new_request_latency")
    block_dir = os.path.join(DATA_DIR, "main/sharegpt/min_new_request_latency")
    llumnix_dir = os.path.join(DATA_DIR, "main/sharegpt/min_lunmnix_load")

    po2_qps, po2_ttft = load_system_capacity_data(po2_dir)
    block_qps, block_ttft = load_system_capacity_data(block_dir, "len_estimated_true")
    llumnix_qps, llumnix_ttft = load_system_capacity_data(llumnix_dir, "len_estimated_false")

    po2_capacity = calculate_capacity(po2_qps, po2_ttft, ttft_slo)
    block_capacity = calculate_capacity(block_qps, block_ttft, ttft_slo)
    llumnix_capacity = calculate_capacity(llumnix_qps, llumnix_ttft, ttft_slo)

    systems = [LABELS['po2'], LABELS['astrolabe'], LABELS['llumnix']]
    capacities = [po2_capacity, block_capacity, llumnix_capacity]
    colors_list = [COLORS['po2'], COLORS['astrolabe'], COLORS['llumnix']]

    x = np.arange(len(systems))
    bars = ax.bar(x, capacities, color=colors_list, edgecolor='black', width=0.6)

    for bar, cap in zip(bars, capacities):
        height = bar.get_height()
        ax.annotate(f'{cap:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Capacity (QPS)', fontsize=12)
    ax.set_title(f'Capacity (TTFT P99 ≤ {ttft_slo}s)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    min_cap = min(capacities) * 0.9
    max_cap = max(capacities) * 1.1
    ax.set_ylim(min_cap, max_cap)

    return {'po2': po2_capacity, 'astrolabe': block_capacity, 'llumnix': llumnix_capacity}


def plot_full_comparison(results, output_dir, ttft_slo=3.0):
    """Plot comprehensive 2x2 comparison: latency, capacity, p99, overhead."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    qps = results['qps']

    plot_single_metric(axes[0, 0], qps,
                       results['po2_mean'], results['block_mean'], results['llumnix_mean'],
                       'Mean E2E Latency (s)', 'Mean Latency vs QPS', show_legend=True)

    capacities = plot_capacity_bar(axes[0, 1], ttft_slo=ttft_slo)

    plot_single_metric(axes[1, 0], qps,
                       results['po2_p99'], results['block_p99'], results['llumnix_p99'],
                       'P99 E2E Latency (s)', 'P99 Latency vs QPS', show_legend=False)

    plot_single_metric(axes[1, 1], qps,
                       results['po2_overhead'], results['block_overhead'], results['llumnix_overhead'],
                       'Scheduling Overhead (ms)', 'Scheduling Overhead vs QPS', show_legend=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'po2_comparison.png'), dpi=150, bbox_inches='tight')
    
    plt.close()
    print(f"Saved: {output_dir}/po2_comparison.png ")

    return capacities


def main():
    parser = argparse.ArgumentParser(description='Plot Po2 comparison for extra ablation')
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR,
                        help='Output directory for plots')
    parser.add_argument('--ttft-slo', type=float, default=3.0,
                        help='TTFT P99 SLO in seconds for capacity calculation')
    parser.add_argument('--system-name', type=str, default="Astrolabe",
                        help="Name of the system (default: Astrolabe, use Astrolabe for anonymous submission)")
    args = parser.parse_args()

    # Update global LABELS based on system name
    global SYSTEM_NAME, LABELS
    SYSTEM_NAME = args.system_name
    LABELS = get_labels(SYSTEM_NAME)

    results = load_results()

    if not results['qps']:
        print("No data found! Check data directories.")
        return

    # Print summary
    print("\n" + "="*90)
    print("Po2 Ablation Results - Scheduling Overhead Comparison")
    print("="*90)
    print(f"{'QPS':<6} {'Po2 (ms)':<14} {'Astrolabe (ms)':<14} {'Llumnix (ms)':<14} {'Po2 vs Astrolabe':<14}")
    print("-" * 90)

    for i, qps in enumerate(results['qps']):
        overhead_reduction = ((results['block_overhead'][i] - results['po2_overhead'][i]) / results['block_overhead'][i]) * 100 if results['block_overhead'][i] > 0 else 0
        print(f"{qps:<6} {results['po2_overhead'][i]:<14.1f} {results['block_overhead'][i]:<14.1f} {results['llumnix_overhead'][i]:<14.1f} {overhead_reduction:+.1f}% reduction")

    # Generate plots
    capacities = plot_full_comparison(results, args.output_dir, ttft_slo=args.ttft_slo)

    print("\n" + "="*90)
    print(f"Capacity (TTFT P99 ≤ {args.ttft_slo}s SLO)")
    print("="*90)
    print(f"{SYSTEM_NAME}-Po2 (N=2): {capacities['po2']:.1f} QPS")
    print(f"{SYSTEM_NAME} (N=12):    {capacities['astrolabe']:.1f} QPS")
    print(f"Llumnix-:       {capacities['llumnix']:.1f} QPS")


if __name__ == '__main__':
    main()