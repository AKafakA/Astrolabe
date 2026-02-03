#!/usr/bin/env python3
"""
Extra Ablation: Plot burstiness study results comparing Astrolabe vs Llumnix-.

Burstiness values (gamma distribution shape parameter):
- 0.25: Most bursty (high variance in inter-arrival times)
- 0.5: Moderately bursty
- 1.0: Poisson arrivals (baseline)
- 2.0: Regular arrivals (low variance)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "experiment_output/data/extra_ablation/burstiness")
RESULTS_DIR = os.path.join(BASE_DIR, "experiment_output/results/extra_ablation")

# Default system name (can be overridden via --system-name argument)
SYSTEM_NAME = "Astrolabe"

def get_colors(system_name="Astrolabe"):
    return {system_name: '#2ecc71', 'Llumnix-': '#3498db'}

# Colors (will be updated in main based on --system-name)
COLORS = get_colors()


def load_results(data_dir, system_name="Astrolabe"):
    """Load all experiment results."""
    burstiness_levels = [0.25, 0.5, 1.0, 2.0]
    schedulers = {
        'min_new_request_latency': system_name,
        'min_lunmnix_load': 'Llumnix-'
    }

    results = {scheduler: {'burstiness': [], 'mean_latency': [], 'p99_latency': [],
                           'throughput': [], 'waiting_latency': []}
               for scheduler in schedulers.values()}

    for burstiness in burstiness_levels:
        dir_name = f'burstiness_{burstiness}'
        for scheduler_code, scheduler_name in schedulers.items():
            scheduler_dir = os.path.join(data_dir, dir_name, 'sharegpt', scheduler_code)
            npz_files = []
            for root, dirs, files in os.walk(scheduler_dir):
                for f in files:
                    if f.endswith('.npz'):
                        npz_files.append(os.path.join(root, f))

            if npz_files:
                data = np.load(npz_files[0])
                e2e = data['request_latencies']
                results[scheduler_name]['burstiness'].append(burstiness)
                results[scheduler_name]['mean_latency'].append(np.mean(e2e))
                results[scheduler_name]['p99_latency'].append(np.percentile(e2e, 99))
                results[scheduler_name]['throughput'].append(float(data['Throughput']))

    return results


def plot_latency_comparison(results, output_dir, system_name="Astrolabe"):
    """Plot mean and P99 latency comparison as grouped bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    burstiness_levels = results[system_name]['burstiness']
    x = np.arange(len(burstiness_levels))
    width = 0.35

    # Mean latency
    block_mean = np.array(results[system_name]['mean_latency']) / 1000
    llumnix_mean = np.array(results['Llumnix-']['mean_latency']) / 1000

    bars1 = axes[0].bar(x - width/2, block_mean, width, label=system_name, color=COLORS[system_name], edgecolor='black')
    bars2 = axes[0].bar(x + width/2, llumnix_mean, width, label='Llumnix-', color=COLORS['Llumnix-'], edgecolor='black')

    axes[0].set_xlabel('Burstiness (gamma shape parameter)', fontsize=12)
    axes[0].set_ylabel('Mean E2E Latency (s)', fontsize=12)
    axes[0].set_title('Mean Latency vs Burstiness', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(b) for b in burstiness_levels])
    axes[0].legend()
    axes[0].set_ylim(0, max(max(block_mean), max(llumnix_mean)) * 1.15)

    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # P99 latency
    block_p99 = np.array(results[system_name]['p99_latency']) / 1000
    llumnix_p99 = np.array(results['Llumnix-']['p99_latency']) / 1000

    bars3 = axes[1].bar(x - width/2, block_p99, width, label=system_name, color=COLORS[system_name], edgecolor='black')
    bars4 = axes[1].bar(x + width/2, llumnix_p99, width, label='Llumnix-', color=COLORS['Llumnix-'], edgecolor='black')

    axes[1].set_xlabel('Burstiness (gamma shape parameter)', fontsize=12)
    axes[1].set_ylabel('P99 E2E Latency (s)', fontsize=12)
    axes[1].set_title('P99 Latency vs Burstiness', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(b) for b in burstiness_levels])
    axes[1].legend()
    axes[1].set_ylim(0, max(max(block_p99), max(llumnix_p99)) * 1.15)

    for bar in bars3:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'burstiness_comparison.png'), dpi=150, bbox_inches='tight')
    
    plt.close()
    print(f"Saved: {output_dir}/burstiness_comparison.png ")


def plot_latency_lines(results, output_dir, system_name="Astrolabe"):
    """Plot latency trends as line chart."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    burstiness_levels = results[system_name]['burstiness']

    # Mean latency
    block_mean = np.array(results[system_name]['mean_latency']) / 1000
    llumnix_mean = np.array(results['Llumnix-']['mean_latency']) / 1000

    axes[0].plot(burstiness_levels, block_mean, 'o-', color=COLORS[system_name], linewidth=2,
                 markersize=10, label=system_name, markeredgecolor='black')
    axes[0].plot(burstiness_levels, llumnix_mean, 's--', color=COLORS['Llumnix-'], linewidth=2,
                 markersize=10, label='Llumnix-', markeredgecolor='black')

    axes[0].set_xlabel('Burstiness (gamma shape parameter)', fontsize=12)
    axes[0].set_ylabel('Mean E2E Latency (s)', fontsize=12)
    axes[0].set_title('Mean Latency vs Burstiness', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(burstiness_levels)

    # P99 latency
    block_p99 = np.array(results[system_name]['p99_latency']) / 1000
    llumnix_p99 = np.array(results['Llumnix-']['p99_latency']) / 1000

    axes[1].plot(burstiness_levels, block_p99, 'o-', color=COLORS[system_name], linewidth=2,
                 markersize=10, label=system_name, markeredgecolor='black')
    axes[1].plot(burstiness_levels, llumnix_p99, 's--', color=COLORS['Llumnix-'], linewidth=2,
                 markersize=10, label='Llumnix-', markeredgecolor='black')

    axes[1].set_xlabel('Burstiness (gamma shape parameter)', fontsize=12)
    axes[1].set_ylabel('P99 E2E Latency (s)', fontsize=12)
    axes[1].set_title('P99 Latency vs Burstiness', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(burstiness_levels)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'burstiness_lines.png'), dpi=150, bbox_inches='tight')
    
    plt.close()
    print(f"Saved: {output_dir}/burstiness_lines.png ")


def main():
    parser = argparse.ArgumentParser(description='Plot burstiness study for extra ablation')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR,
                        help='Output directory for plots')
    parser.add_argument('--system-name', type=str, default="Astrolabe",
                        help="Name of the system (default: Astrolabe, use Astrolabe for anonymous submission)")
    args = parser.parse_args()

    # Update global COLORS based on system name
    global SYSTEM_NAME, COLORS
    SYSTEM_NAME = args.system_name
    COLORS = get_colors(SYSTEM_NAME)

    results = load_results(args.data_dir, SYSTEM_NAME)

    if not results[SYSTEM_NAME]['burstiness']:
        print("No data found! Check data directories.")
        return

    # Print summary
    print("\n=== Burstiness Study Results ===\n")

    print(f"{SYSTEM_NAME} Results:")
    print(f"{'Burstiness':<12} {'Mean (ms)':<12} {'P99 (ms)':<12} {'Throughput':<12}")
    print("-" * 48)
    for i, b in enumerate(results[SYSTEM_NAME]['burstiness']):
        print(f"{b:<12} {results[SYSTEM_NAME]['mean_latency'][i]:<12.0f} {results[SYSTEM_NAME]['p99_latency'][i]:<12.0f} {results[SYSTEM_NAME]['throughput'][i]:<12.1f}")

    print("\nLlumnix- Results:")
    print(f"{'Burstiness':<12} {'Mean (ms)':<12} {'P99 (ms)':<12} {'Throughput':<12}")
    print("-" * 48)
    for i, b in enumerate(results['Llumnix-']['burstiness']):
        print(f"{b:<12} {results['Llumnix-']['mean_latency'][i]:<12.0f} {results['Llumnix-']['p99_latency'][i]:<12.0f} {results['Llumnix-']['throughput'][i]:<12.1f}")

    print(f"\n=== {SYSTEM_NAME} vs Llumnix- Comparison ===")
    print(f"{'Burstiness':<12} {'Mean Diff':<12} {'P99 Diff':<12} {'Status':<12}")
    print("-" * 48)
    for i, b in enumerate(results[SYSTEM_NAME]['burstiness']):
        mean_diff = ((results[SYSTEM_NAME]['mean_latency'][i] - results['Llumnix-']['mean_latency'][i]) /
                     results['Llumnix-']['mean_latency'][i]) * 100
        p99_diff = ((results[SYSTEM_NAME]['p99_latency'][i] - results['Llumnix-']['p99_latency'][i]) /
                    results['Llumnix-']['p99_latency'][i]) * 100
        status = "BETTER" if mean_diff < 0 else "WORSE"
        print(f"{b:<12} {mean_diff:+.1f}%{'':<6} {p99_diff:+.1f}%{'':<6} {status:<12}")

    # Generate plot
    plot_latency_lines(results, args.output_dir, SYSTEM_NAME)

    print("\nDone!")


if __name__ == '__main__':
    main()