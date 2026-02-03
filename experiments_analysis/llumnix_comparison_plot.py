"""
Llumnix Comparison Plot for Extra Ablation

Generates comparison plots between Astrolabe and Llumnix (Migration) from collected results.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Default system name (can be overridden via --system-name argument)
SYSTEM_NAME = "Astrolabe"

def get_colors(system_name="Astrolabe"):
    return {
        system_name: 'black',
        'Llumnix': '#E24A33',  # Red-orange for Llumnix
    }

def get_markers(system_name="Astrolabe"):
    return {
        system_name: 'o',
        'Llumnix': 's',
    }

# Color scheme matching existing plots (will be updated in main)
COLORS = get_colors()
MARKERS = get_markers()


def load_results_from_logs(base_dir, system_name="Astrolabe"):
    """Load results from log files in the llumnix_compare directory."""
    results = {
        system_name: {},
        'Llumnix': {}
    }

    # Load Astrolabe results
    block_dir = os.path.join(base_dir, 'block')
    if os.path.exists(block_dir):
        for qps_folder in os.listdir(block_dir):
            if qps_folder.startswith('qps'):
                qps = int(qps_folder.replace('qps', ''))
                qps_path = os.path.join(block_dir, qps_folder)
                for f in os.listdir(qps_path):
                    if f.endswith('_logs.txt'):
                        log_path = os.path.join(qps_path, f)
                        results[system_name][qps] = parse_log_file(log_path)
                    if f.endswith('.npz'):
                        npz_path = os.path.join(qps_path, f)
                        npz_data = np.load(npz_path)
                        if qps in results[system_name]:
                            results[system_name][qps]['npz'] = npz_data

    # Load Llumnix results
    llumnix_dir = os.path.join(base_dir, 'llumnix')
    if os.path.exists(llumnix_dir):
        for qps_folder in os.listdir(llumnix_dir):
            if qps_folder.startswith('qps'):
                qps = int(qps_folder.replace('qps', ''))
                qps_path = os.path.join(llumnix_dir, qps_folder)
                for f in os.listdir(qps_path):
                    if f.endswith('_logs.txt') or f.endswith('logs.txt'):
                        log_path = os.path.join(qps_path, f)
                        results['Llumnix'][qps] = parse_log_file(log_path)
                    if f.endswith('.npz'):
                        npz_path = os.path.join(qps_path, f)
                        npz_data = np.load(npz_path)
                        if qps in results['Llumnix']:
                            results['Llumnix'][qps]['npz'] = npz_data

    return results


def parse_log_file(log_path):
    """Parse a benchmark log file to extract metrics."""
    metrics = {}
    with open(log_path, 'r') as f:
        content = f.read()

        # Parse throughput
        if 'tokens_per_s' in content:
            import re
            match = re.search(r'tokens_per_s\s+(\d+\.?\d*)', content)
            if match:
                metrics['throughput'] = float(match.group(1))

        # Parse mean token latency
        match = re.search(r'mean_token_latency=(\d+\.?\d*)', content)
        if match:
            metrics['mean_token_latency'] = float(match.group(1))

        # Parse mean e2e latency
        match = re.search(r'mean_e2e_latency=(\d+\.?\d*)', content)
        if match:
            metrics['mean_e2e_latency'] = float(match.group(1))

        # Parse scheduling overhead
        match = re.search(r'mean_global_scheduling_overhead=(\d+\.?\d*)', content)
        if match:
            metrics['scheduling_overhead'] = float(match.group(1))

        # Parse p99 request latency
        match = re.search(r'p99 request latency:\s*(\d+\.?\d*)', content)
        if match:
            metrics['p99_e2e_latency'] = float(match.group(1))

        # Parse p99 prefill latency
        match = re.search(r'p99 prefill token latency:\s*(\d+\.?\d*)', content)
        if match:
            metrics['p99_prefill_latency'] = float(match.group(1))

    return metrics


def plot_comparison(results, output_dir, system_name="Astrolabe"):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Get common QPS values
    block_qps = sorted(results[system_name].keys())
    llumnix_qps = sorted(results['Llumnix'].keys())
    common_qps = sorted(set(block_qps) & set(llumnix_qps))

    if not common_qps:
        print("No common QPS values found!")
        return

    print(f"Plotting for QPS values: {common_qps}")

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Mean E2E Latency
    ax = axes[0, 0]
    for system in [system_name, 'Llumnix']:
        qps_vals = []
        latency_vals = []
        for qps in common_qps:
            if qps in results[system] and 'mean_e2e_latency' in results[system][qps]:
                qps_vals.append(qps)
                latency_vals.append(results[system][qps]['mean_e2e_latency'] / 1000)  # Convert to seconds
        ax.plot(qps_vals, latency_vals, marker=MARKERS[system], color=COLORS[system],
                label=system, linewidth=2, markersize=8)
    ax.set_xlabel('QPS', fontsize=12)
    ax.set_ylabel('Mean E2E Latency (s)', fontsize=12)
    ax.set_title('Mean End-to-End Latency', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to show the difference at high QPS

    # Plot 2: Throughput
    ax = axes[0, 1]
    for system in [system_name, 'Llumnix']:
        qps_vals = []
        throughput_vals = []
        for qps in common_qps:
            if qps in results[system] and 'throughput' in results[system][qps]:
                qps_vals.append(qps)
                throughput_vals.append(results[system][qps]['throughput'])
        ax.plot(qps_vals, throughput_vals, marker=MARKERS[system], color=COLORS[system],
                label=system, linewidth=2, markersize=8)
    ax.set_xlabel('QPS', fontsize=12)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax.set_title('Token Throughput', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Mean Token Latency
    ax = axes[1, 0]
    for system in [system_name, 'Llumnix']:
        qps_vals = []
        latency_vals = []
        for qps in common_qps:
            if qps in results[system] and 'mean_token_latency' in results[system][qps]:
                qps_vals.append(qps)
                latency_vals.append(results[system][qps]['mean_token_latency'])
        ax.plot(qps_vals, latency_vals, marker=MARKERS[system], color=COLORS[system],
                label=system, linewidth=2, markersize=8)
    ax.set_xlabel('QPS', fontsize=12)
    ax.set_ylabel('Mean Token Latency (ms)', fontsize=12)
    ax.set_title('Mean Token Latency', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale

    # Plot 4: P99 E2E Latency
    ax = axes[1, 1]
    for system in [system_name, 'Llumnix']:
        qps_vals = []
        latency_vals = []
        for qps in common_qps:
            if qps in results[system] and 'p99_e2e_latency' in results[system][qps]:
                qps_vals.append(qps)
                latency_vals.append(results[system][qps]['p99_e2e_latency'] / 1000)  # Convert to seconds
        ax.plot(qps_vals, latency_vals, marker=MARKERS[system], color=COLORS[system],
                label=system, linewidth=2, markersize=8)
    ax.set_xlabel('QPS', fontsize=12)
    ax.set_ylabel('P99 E2E Latency (s)', fontsize=12)
    ax.set_title('P99 End-to-End Latency', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llumnix_comparison.png'), bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/llumnix_comparison.png")
    plt.close()

    # Print summary table
    print("\n" + "="*80)
    print(f"{system_name} vs Llumnix (Migration) Comparison Summary")
    print("="*80)
    print(f"{'QPS':<6} | {f'{system_name} E2E (s)':<14} | {'Llumnix E2E (s)':<16} | {f'{system_name} Advantage':<16}")
    print("-"*80)
    for qps in common_qps:
        block_e2e = results[system_name].get(qps, {}).get('mean_e2e_latency', 0) / 1000
        llumnix_e2e = results['Llumnix'].get(qps, {}).get('mean_e2e_latency', 0) / 1000
        if block_e2e > 0 and llumnix_e2e > 0:
            advantage = llumnix_e2e / block_e2e
            print(f"{qps:<6} | {block_e2e:<14.2f} | {llumnix_e2e:<16.2f} | {advantage:<16.1f}x")
    print("="*80)


def plot_bar_comparison(results, output_dir, system_name="Astrolabe"):
    """Generate bar chart comparison for key QPS points."""
    os.makedirs(output_dir, exist_ok=True)

    # Select key QPS points: low (16), medium (28), high (36)
    key_qps = [16, 28, 36]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ('mean_e2e_latency', 'Mean E2E Latency (s)', 1000),  # divisor
        ('throughput', 'Throughput (tok/s)', 1),
        ('mean_token_latency', 'Mean Token Latency (ms)', 1),
    ]

    for idx, (metric, ylabel, divisor) in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(key_qps))
        width = 0.35

        block_vals = []
        llumnix_vals = []

        for qps in key_qps:
            block_val = results[system_name].get(qps, {}).get(metric, 0) / divisor
            llumnix_val = results['Llumnix'].get(qps, {}).get(metric, 0) / divisor
            block_vals.append(block_val)
            llumnix_vals.append(llumnix_val)

        bars1 = ax.bar(x - width/2, block_vals, width, label=system_name, color=COLORS[system_name])
        bars2 = ax.bar(x + width/2, llumnix_vals, width, label='Llumnix', color=COLORS['Llumnix'])

        ax.set_xlabel('QPS', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(q) for q in key_qps])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'llumnix_comparison_bars.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'llumnix_comparison_bars.png'), bbox_inches='tight', dpi=300)
    print(f"Saved bar plot to {output_path}")
    plt.close()


def main():
    # Base paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(base_path, 'experiment_output/data/extra_ablation/llumnix_compare')
    default_output_dir = os.path.join(base_path, 'experiment_output/results/extra_ablation')

    parser = argparse.ArgumentParser(description='Plot Astrolabe vs Llumnix comparison')
    parser.add_argument('--data-dir', type=str,
                        default=default_data_dir,
                        help='Directory containing comparison results')
    parser.add_argument('--output-dir', type=str,
                        default=default_output_dir,
                        help='Output directory for plots')
    parser.add_argument('--system-name', type=str, default="Astrolabe",
                        help="Name of the system (default: Astrolabe, use Astrolabe for anonymous submission)")
    args = parser.parse_args()

    # Update global settings based on system name
    global SYSTEM_NAME, COLORS, MARKERS
    SYSTEM_NAME = args.system_name
    COLORS = get_colors(SYSTEM_NAME)
    MARKERS = get_markers(SYSTEM_NAME)

    data_dir = args.data_dir
    output_dir = args.output_dir

    print(f"Loading results from: {data_dir}")
    results = load_results_from_logs(data_dir, SYSTEM_NAME)

    print(f"{SYSTEM_NAME} QPS: {sorted(results[SYSTEM_NAME].keys())}")
    print(f"Llumnix QPS: {sorted(results['Llumnix'].keys())}")

    print(f"\nGenerating plots to: {output_dir}")
    plot_comparison(results, output_dir, SYSTEM_NAME)

    print("\nDone!")


if __name__ == "__main__":
    main()
