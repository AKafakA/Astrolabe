#!/usr/bin/env python3
"""
Extra Ablation: Plot CPU Overhead - Astrolabe vs Astrolabe-len-oracle

Shows: average CPU utilization per process and total CPU cores used.
Astrolabe = with length estimation, Astrolabe-len-oracle = without length estimation.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "experiment_output/data/extra_ablation/cpu_tracker/sharegpt/min_new_request_latency")
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_output/results/extra_ablation")

# Plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (12, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
})

NUM_PREDICTORS = 16
TOTAL_CORES = 64

# Default system name (can be overridden via --system-name argument)
SYSTEM_NAME = "Astrolabe"

def get_colors(system_name="Astrolabe"):
    return {system_name: '#2ca02c', f'{system_name}-len-oracle': '#d62728'}

def get_markers(system_name="Astrolabe"):
    return {system_name: 's', f'{system_name}-len-oracle': 'o'}

# Colors and markers (will be updated in main based on --system-name)
COLORS = get_colors()
MARKERS = get_markers()


def load_results(data_dir, system_name="Astrolabe"):
    """Load CPU overhead results."""
    experiments = [
        (system_name, "true", [20, 24, 28, 32, 36]),
        (f"{system_name}-len-oracle", "false", [20, 24, 28, 32, 36]),
    ]

    results = {system_name: {}, f'{system_name}-len-oracle': {}}

    for name, len_est, qps_list in experiments:
        for qps in qps_list:
            pattern = f"qps_{qps}_num_queries_10000_n_12_chunked_true_predictor_16_global_1_len_estimated_{len_est}_max_slo_0_enable_preemptive_auto_provisioning_false_batch_48_chunk_512"
            npz_path = os.path.join(data_dir, pattern, "benchmark_all_metrics.npz")

            if os.path.exists(npz_path):
                npz = np.load(npz_path)
                if 'avg_predictor_cpu_percent' in npz.files:
                    avg_cpu = np.mean(npz['avg_predictor_cpu_percent'])
                    results[name][qps] = {
                        'avg_cpu_per_process': avg_cpu,
                        'total_cpu_percent': avg_cpu * NUM_PREDICTORS,
                        'total_cores': avg_cpu * NUM_PREDICTORS / 100,
                    }
            else:
                print(f"File not found: {npz_path}")

    return results


def plot_cpu_overhead(results, output_dir, system_name="Astrolabe"):
    """Generate CPU overhead plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    system_oracle = f'{system_name}-len-oracle'

    # Plot 1: Average CPU Utilization per Process
    ax1 = axes[0]
    for name in [system_name, system_oracle]:
        data = results[name]
        qps_list = sorted(data.keys())
        values = [data[q]['avg_cpu_per_process'] for q in qps_list]
        ax1.plot(qps_list, values, marker=MARKERS[name], color=COLORS[name],
                 label=name, linewidth=2.5, markersize=10)

    ax1.set_xlabel('QPS')
    ax1.set_ylabel('CPU Utilization per Predictor (%)')
    ax1.set_title('Average CPU Utilization per Predictor Process')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Plot 2: Total CPU Cores Used
    ax2 = axes[1]
    for name in [system_name, system_oracle]:
        data = results[name]
        qps_list = sorted(data.keys())
        values = [data[q]['total_cores'] for q in qps_list]
        ax2.plot(qps_list, values, marker=MARKERS[name], color=COLORS[name],
                 label=name, linewidth=2.5, markersize=10)

    ax2.set_xlabel('QPS')
    ax2.set_ylabel('Total CPU Cores Used')
    ax2.set_title(f'Total CPU Cores Used (out of {TOTAL_CORES} cores)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax2.axhline(y=TOTAL_CORES, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, max(15, max([d['total_cores'] for d in results[system_oracle].values()]) * 1.1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_overhead.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/cpu_overhead.png")


def main():
    parser = argparse.ArgumentParser(description='Plot CPU overhead for extra ablation')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR,
                        help='Output directory for plots')
    parser.add_argument('--system-name', type=str, default="Astrolabe",
                        help="Name of the system (default: Astrolabe, use Astrolabe for anonymous submission)")
    args = parser.parse_args()

    # Update global settings based on system name
    global SYSTEM_NAME, COLORS, MARKERS
    SYSTEM_NAME = args.system_name
    COLORS = get_colors(SYSTEM_NAME)
    MARKERS = get_markers(SYSTEM_NAME)

    system_oracle = f'{SYSTEM_NAME}-len-oracle'

    results = load_results(args.data_dir, SYSTEM_NAME)

    print(f"{SYSTEM_NAME} data: {sorted(results[SYSTEM_NAME].keys())}")
    print(f"{system_oracle} data: {sorted(results[system_oracle].keys())}")

    if not results[SYSTEM_NAME] or not results[system_oracle]:
        print("No data found! Check data directories.")
        return

    plot_cpu_overhead(results, args.output_dir, SYSTEM_NAME)

    # Print summary
    print("\n" + "="*90)
    print("CPU Overhead Summary")
    print("="*90)
    print(f"{'Scheduler':<20} {'QPS':<6} {'CPU/Predictor':<15} {'Total CPU %':<15} {'Cores Used':<12} {'% of 64':<10}")
    print("-"*90)

    for name in [SYSTEM_NAME, system_oracle]:
        for qps in sorted(results[name].keys()):
            data = results[name][qps]
            pct_of_total = data['total_cores'] / TOTAL_CORES * 100
            print(f"{name:<20} {qps:<6} {data['avg_cpu_per_process']:>12.1f}% {data['total_cpu_percent']:>13.1f}% {data['total_cores']:>10.1f} {pct_of_total:>9.1f}%")

    print("\n" + "="*90)
    print(f"{SYSTEM_NAME} = with length estimation | {system_oracle} = without length estimation")
    print(f"System: {TOTAL_CORES} CPU cores | Predictors: {NUM_PREDICTORS}")
    print("="*90)


if __name__ == '__main__':
    main()