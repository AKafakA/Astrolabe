#!/usr/bin/env python3
"""
Extra Ablation: Plot heatmaps showing latency degradation percentage for prediction error sensitivity.

Deeper color indicates more degradation compared to baseline (0%, 0%).
Also shows comparison with Llumnix- baseline to demonstrate robustness.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "experiment_output/data/extra_ablation/prediction_error")
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_output/results/extra_ablation")


def load_results(data_dir):
    """Load all experiment results and compute metrics."""
    len_errors = [0, 25, 50, 100]
    lat_errors = [0, 25, 50, 100]

    mean_latency = np.zeros((len(len_errors), len(lat_errors)))
    p99_latency = np.zeros((len(len_errors), len(lat_errors)))
    throughput = np.zeros((len(len_errors), len(lat_errors)))

    for i, len_err in enumerate(len_errors):
        for j, lat_err in enumerate(lat_errors):
            dir_name = f'len_err_{len_err}_lat_err_{lat_err}'
            npz_files = []
            for root, dirs, files in os.walk(os.path.join(data_dir, dir_name)):
                if 'min_new_request_latency' in root:
                    for f in files:
                        if f.endswith('.npz'):
                            npz_files.append(os.path.join(root, f))

            if npz_files:
                data = np.load(npz_files[0])
                e2e = data['request_latencies']
                mean_latency[i, j] = np.mean(e2e)
                p99_latency[i, j] = np.percentile(e2e, 99)
                throughput[i, j] = float(data['Throughput'])

    return len_errors, lat_errors, mean_latency, p99_latency, throughput


def load_baseline(data_dir):
    """Load Llumnix- baseline results."""
    baseline_dir = os.path.join(data_dir, 'len_err_0_lat_err_0/sharegpt/min_lunmnix_load')
    npz_files = []
    for root, dirs, files in os.walk(baseline_dir):
        for f in files:
            if f.endswith('.npz'):
                npz_files.append(os.path.join(root, f))

    if npz_files:
        data = np.load(npz_files[0])
        e2e = data['request_latencies']
        return {
            'mean': np.mean(e2e),
            'p99': np.percentile(e2e, 99),
            'throughput': float(data['Throughput'])
        }
    return None


def compute_degradation(matrix):
    """Compute percentage degradation relative to baseline (0%, 0%)."""
    baseline = matrix[0, 0]
    degradation = ((matrix - baseline) / baseline) * 100
    return degradation


def plot_combined_heatmaps(mean_lat, p99_lat, baseline, len_errors, lat_errors, output_dir, show_baseline_line=True):
    """Plot mean and P99 heatmaps with baseline comparison."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    block_baseline_mean = mean_lat[0, 0]
    block_baseline_p99 = p99_lat[0, 0]

    mean_deg = ((mean_lat - block_baseline_mean) / block_baseline_mean) * 100
    p99_deg = ((p99_lat - block_baseline_p99) / block_baseline_p99) * 100

    llumnix_mean_deg = ((baseline['mean'] - block_baseline_mean) / block_baseline_mean) * 100 if baseline else 0
    llumnix_p99_deg = ((baseline['p99'] - block_baseline_p99) / block_baseline_p99) * 100 if baseline else 0

    if baseline:
        print(f"Llumnix- baseline - Mean deg: {llumnix_mean_deg:.2f}%, P99 deg: {llumnix_p99_deg:.2f}%")

    # Mean latency degradation
    vmax_mean = max(3, mean_deg.max(), llumnix_mean_deg + 0.5) if baseline else max(3, mean_deg.max())
    im1 = sns.heatmap(mean_deg,
                annot=True,
                fmt='.1f',
                cmap='Reds',
                xticklabels=[f'{e}%' for e in lat_errors],
                yticklabels=[f'{e}%' for e in len_errors],
                ax=axes[0],
                vmin=0,
                vmax=vmax_mean,
                annot_kws={'size': 11, 'weight': 'bold'},
                cbar_kws={'label': 'Degradation (%)'})
    axes[0].set_xlabel('Latency Prediction Error', fontsize=12)
    axes[0].set_ylabel('Length Prediction Error', fontsize=12)
    axes[0].set_title('Mean Latency Degradation (%)', fontsize=13)

    if show_baseline_line and baseline:
        cbar1 = im1.collections[0].colorbar
        cbar1.ax.axhline(y=llumnix_mean_deg, color='blue', linewidth=2.5, linestyle='-')
        cbar1.ax.text(1.5, llumnix_mean_deg, f' Llumnix- ({llumnix_mean_deg:.1f}%)',
                      fontsize=9, va='center', ha='left', color='blue', fontweight='bold')

    # P99 latency degradation
    vmax_p99 = max(4, p99_deg.max(), llumnix_p99_deg + 0.5) if baseline else max(4, p99_deg.max())
    im2 = sns.heatmap(p99_deg,
                annot=True,
                fmt='.1f',
                cmap='Reds',
                xticklabels=[f'{e}%' for e in lat_errors],
                yticklabels=[f'{e}%' for e in len_errors],
                ax=axes[1],
                vmin=0,
                vmax=vmax_p99,
                annot_kws={'size': 11, 'weight': 'bold'},
                cbar_kws={'label': 'Degradation (%)'})
    axes[1].set_xlabel('Latency Prediction Error', fontsize=12)
    axes[1].set_ylabel('Length Prediction Error', fontsize=12)
    axes[1].set_title('P99 Latency Degradation (%)', fontsize=13)

    if show_baseline_line and baseline:
        cbar2 = im2.collections[0].colorbar
        cbar2.ax.axhline(y=llumnix_p99_deg, color='blue', linewidth=2.5, linestyle='-')
        cbar2.ax.text(1.5, llumnix_p99_deg - 0.3, f' Llumnix- ({llumnix_p99_deg:.1f}%)',
                      fontsize=9, va='top', ha='left', color='blue', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_error_heatmap.png'), dpi=150, bbox_inches='tight')
    
    plt.close()
    print(f"Saved: {output_dir}/prediction_error_heatmap.png ")


def plot_comparison_bar(mean_lat, p99_lat, baseline, len_errors, lat_errors, output_dir):
    """Plot bar chart comparing Astrolabe variants with Llumnix baseline."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ['Astrolabe\n(0%,0%)', 'Astrolabe\n(25%,25%)', 'Astrolabe\n(50%,50%)', 'Astrolabe\n(100%,100%)', 'Llumnix-']
    mean_values = [mean_lat[0,0], mean_lat[1,1], mean_lat[2,2], mean_lat[3,3], baseline['mean']]
    p99_values = [p99_lat[0,0], p99_lat[1,1], p99_lat[2,2], p99_lat[3,3], baseline['p99']]

    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#3498db']

    # Mean latency
    bars1 = axes[0].bar(labels, mean_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Mean Latency (ms)', fontsize=12)
    axes[0].set_title('Mean Latency Comparison', fontsize=13)
    axes[0].set_ylim(0, max(mean_values) * 1.15)
    for bar, val in zip(bars1, mean_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    # P99 latency
    bars2 = axes[1].bar(labels, p99_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('P99 Latency (ms)', fontsize=12)
    axes[1].set_title('P99 Latency Comparison', fontsize=13)
    axes[1].set_ylim(0, max(p99_values) * 1.15)
    for bar, val in zip(bars2, p99_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_error_bar.png'), dpi=150, bbox_inches='tight')
    
    plt.close()
    print(f"Saved: {output_dir}/prediction_error_bar.png ")


def main():
    parser = argparse.ArgumentParser(description='Plot prediction error heatmaps for extra ablation')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR,
                        help='Output directory for plots')
    parser.add_argument('--no-baseline-line', action='store_true',
                        help='Do not show Llumnix baseline line on colorbar')
    args = parser.parse_args()

    len_errors, lat_errors, mean_latency, p99_latency, throughput = load_results(args.data_dir)
    baseline = load_baseline(args.data_dir)

    if mean_latency[0, 0] == 0:
        print("No data found! Check data directories.")
        return

    # Compute degradation
    mean_deg = compute_degradation(mean_latency)
    p99_deg = compute_degradation(p99_latency)

    print("\n=== Astrolabe Results ===")
    print(f"Astrolabe baseline (0%,0%): Mean={mean_latency[0,0]:.0f}ms, P99={p99_latency[0,0]:.0f}ms")

    if baseline:
        print(f"\n=== Llumnix- Baseline ===")
        print(f"Mean={baseline['mean']:.0f}ms, P99={baseline['p99']:.0f}ms, Throughput={baseline['throughput']:.1f}")

        print("\n=== Astrolabe vs Llumnix- Comparison ===")
        for i, len_err in enumerate(len_errors):
            for j, lat_err in enumerate(lat_errors):
                mean_diff = ((mean_latency[i,j] - baseline['mean']) / baseline['mean']) * 100
                p99_diff = ((p99_latency[i,j] - baseline['p99']) / baseline['p99']) * 100
                status = "BETTER" if mean_latency[i,j] < baseline['mean'] else "WORSE"
                print(f"  Astrolabe({len_err}%,{lat_err}%): Mean {mean_diff:+.1f}%, P99 {p99_diff:+.1f}% vs Llumnix- [{status}]")

    print("\n=== Mean Latency Degradation (%) ===")
    print(mean_deg)

    print("\n=== P99 Latency Degradation (%) ===")
    print(p99_deg)

    # Generate plot
    plot_combined_heatmaps(mean_latency, p99_latency, baseline, len_errors, lat_errors,
                           args.output_dir, show_baseline_line=not args.no_baseline_line)

    print("\nDone!")


if __name__ == '__main__':
    main()