#!/usr/bin/env python3
"""
Visualize detailed memory breakdown for DP-SGD experiments
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


EXPERIMENT_ORDER = ["vanilla", "ghost", "flash_clip", "flash_clip_bookkeeping", "bookkeeping"]

NAME_MAPPING = {
    "vanilla": "Classical Training / No DP",
    "ghost": "SOTA DP Training",
    "flash_clip": "FlashClip",
    "flash_clip_bookkeeping": "FlashClip w/ Bookkeeping",
    "bookkeeping": "Bookkeeping",
}

COLORS = {
    "vanilla": "lightblue",
    "ghost": "lightcoral",
    "flash_clip": "lightgreen",
    "flash_clip_bookkeeping": "lightyellow",
    "bookkeeping": "lightpink",
}


def load_results(input_dir):
    """Load all experiment results"""
    input_path = Path(input_dir)
    results = {}
    
    for exp_name in EXPERIMENT_ORDER:
        result_file = input_path / f"{exp_name}_result.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[exp_name] = json.load(f)
            print(f"✓ Loaded {exp_name} results")
        else:
            print(f"⚠️  Missing {exp_name} results")
    
    return results


def plot_memory_breakdown_comparison(results, output_dir):
    """Create detailed memory breakdown comparison"""
    
    # Extract breakdown data
    experiments = []
    breakdowns = {}
    
    for exp_key in EXPERIMENT_ORDER:
        if exp_key in results:
            experiments.append(NAME_MAPPING[exp_key])
            breakdowns[exp_key] = results[exp_key]["breakdown"]
    
    # Define components to track
    components = [
        ("model_parameters_mb", "Model Parameters", "#3498db"),
        ("optimizer_states_mb", "Optimizer States", "#e74c3c"),
        ("gradients_mb", "Gradients", "#f39c12"),
        ("activation_hooks_mb", "Activation Hooks\n(DP-SGD)", "#9b59b6"),
        ("norm_samples_mb", "Norm Samples\n(DP-SGD)", "#1abc9c"),
        ("temp_matrices_mb", "Temp Matrices\n(ggT/aaT)", "#34495e"),
    ]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # ============= Plot 1: Stacked Bar Chart =============
    x = np.arange(len(experiments))
    width = 0.6
    
    # Stack bars
    bottoms = np.zeros(len(experiments))
    
    for comp_key, comp_name, color in components:
        values = []
        for exp_key in EXPERIMENT_ORDER:
            if exp_key in results:
                value = breakdowns[exp_key].get(comp_key, 0)
                values.append(value)
        
        if sum(values) > 0:  # Only plot if non-zero
            ax1.bar(x, values, width, label=comp_name, bottom=bottoms, color=color)
            
            # Add value labels on bars
            for i, (val, bottom) in enumerate(zip(values, bottoms)):
                if val > 500:  # Only label if > 500 MB
                    ax1.text(i, bottom + val/2, f'{val:.0f}', 
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            color='white')
            
            bottoms += np.array(values)
    
    # Add peak memory line
    peak_values = [breakdowns[exp_key].get("peak_allocated_mb", 0) 
                   for exp_key in EXPERIMENT_ORDER if exp_key in results]
    ax1.plot(x, peak_values, 'r*-', markersize=15, linewidth=2, label='Peak Memory', zorder=10)
    
    # Add peak memory labels
    for i, peak in enumerate(peak_values):
        ax1.text(i, peak + 1000, f'{peak:.0f} MB', 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='red')
    
    ax1.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Memory (MB)', fontsize=14, fontweight='bold')
    ax1.set_title('Detailed Memory Breakdown by Component', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments, fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # ============= Plot 2: DP-SGD Overhead Analysis =============
    
    # Calculate DP-SGD overhead (Ghost and Flash Clip vs Vanilla)
    if "vanilla" in results:
        vanilla_peak = breakdowns["vanilla"].get("peak_allocated_mb", 0)
        
        dp_methods = []
        overhead_components = []
        
        for exp_key, exp_name in [("ghost", "Ghost\nClipping"), ("flash_clip", "Flash\nClipping"), ("flash_clip_bookkeeping", "FlashClip w/\nBookkeeping"), ("bookkeeping", "Bookkeeping")]:
            if exp_key in results:
                dp_methods.append(exp_name)
                
                # Calculate overhead for each component
                overhead = {}
                for comp_key, comp_name, _ in components:
                    vanilla_val = breakdowns["vanilla"].get(comp_key, 0)
                    dp_val = breakdowns[exp_key].get(comp_key, 0)
                    overhead[comp_name] = dp_val - vanilla_val
                
                overhead_components.append(overhead)
        
        if dp_methods:
            x2 = np.arange(len(dp_methods))
            width2 = 0.5
            bottoms2 = np.zeros(len(dp_methods))
            
            # Stack overhead bars
            for comp_key, comp_name, color in components:
                values = [overhead.get(comp_name, 0) for overhead in overhead_components]
                
                if sum(values) > 0:
                    ax2.bar(x2, values, width2, label=comp_name, bottom=bottoms2, color=color)
                    
                    # Add labels
                    for i, (val, bottom) in enumerate(zip(values, bottoms2)):
                        if val > 200:
                            ax2.text(i, bottom + val/2, f'{val:.0f}', 
                                    ha='center', va='center', fontsize=9, fontweight='bold',
                                    color='white')
                    
                    bottoms2 += np.array(values)
            
            # Add total overhead labels
            for i, overhead_dict in enumerate(overhead_components):
                total_overhead = sum(overhead_dict.values())
                ax2.text(i, bottoms2[i] + 500, f'+{total_overhead:.0f} MB\n(+{total_overhead/vanilla_peak*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkred')
            
            ax2.set_xlabel('DP-SGD Method', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Additional Memory (MB) vs Vanilla', fontsize=14, fontweight='bold')
            ax2.set_title('DP-SGD Memory Overhead Breakdown', fontsize=16, fontweight='bold')
            ax2.set_xticks(x2)
            ax2.set_xticklabels(dp_methods, fontsize=11)
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "memory_breakdown_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_memory_timeline(results, output_dir):
    """Plot memory usage over time (snapshots)"""
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    
    for idx, (exp_key, ax) in enumerate(zip(EXPERIMENT_ORDER, axes)):
        if exp_key not in results:
            ax.set_visible(False) # Hide unused subplots
            continue
        
        snapshots = results[exp_key]["snapshots"]
        
        # Extract data
        names = [s["name"] for s in snapshots]
        allocated = [s["allocated_mb"] for s in snapshots]
        reserved = [s["reserved_mb"] for s in snapshots]
        
        x = np.arange(len(names))
        
        # Plot
        ax.plot(x, allocated, 'o-', linewidth=2, markersize=8, label='Allocated', color='#3498db')
        ax.plot(x, reserved, 's--', linewidth=2, markersize=6, label='Reserved (Pool)', color='#e74c3c')
        ax.fill_between(x, allocated, reserved, alpha=0.2, color='#e74c3c')
        
        # Highlight key stages
        for i, name in enumerate(names):
            if 'forward' in name:
                ax.axvline(x=i, color='green', linestyle=':', alpha=0.3)
            elif 'backward' in name:
                ax.axvline(x=i, color='orange', linestyle=':', alpha=0.3)
            elif 'step' in name:
                ax.axvline(x=i, color='blue', linestyle=':', alpha=0.3)
        
        ax.set_title(f"{NAME_MAPPING[exp_key]}", fontsize=14, fontweight='bold')
        ax.set_ylabel('Memory (MB)', fontsize=12)
        ax.set_xticks(x[::max(1, len(x)//15)])  # Show every nth tick
        ax.set_xticklabels([names[i] for i in x[::max(1, len(x)//15)]], 
                          rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add peak annotation
        peak_idx = np.argmax(allocated)
        ax.annotate(f'Peak: {allocated[peak_idx]:.0f} MB', 
                   xy=(peak_idx, allocated[peak_idx]), 
                   xytext=(peak_idx, allocated[peak_idx] + 3000),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=11, fontweight='bold', color='red')
    
    axes[-1].set_xlabel('Timeline (Snapshot)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "memory_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_performance_comparison(results, output_dir):
    """Plot time vs memory trade-off"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    name_mapping = {
        "vanilla": "Classical Training / No DP",
        "ghost": "SOTA DP Training",
        "flash_clip": "FlashClip",
        "flash_clip_bookkeeping": "FlashClip w/ Bookkeeping",
        "bookkeeping": "Bookkeeping",
    }
    
    colors = {
        "vanilla": "lightblue",
        "ghost": "lightcoral",
        "flash_clip": "lightgreen",
        "flash_clip_bookkeeping": "lightyellow",
        "bookkeeping": "lightpink",
    }
    
    all_peak_mems = []
    all_avg_times = []

    for exp_key in EXPERIMENT_ORDER:
        if exp_key not in results:
            continue
        
        peak_mem = results[exp_key]["peak_allocated_memory_mb"] # Use allocated memory
        avg_time = results[exp_key]["avg_time_ms"]
        all_peak_mems.append(peak_mem)
        all_avg_times.append(avg_time)
        
        ax.scatter(peak_mem, avg_time, s=500, alpha=0.7, 
                  color=COLORS[exp_key].replace("light", ""), edgecolors='black', linewidth=2,
                  label=NAME_MAPPING[exp_key])
        
        # Add labels
        ax.annotate(NAME_MAPPING[exp_key], 
                   xy=(peak_mem, avg_time), 
                   xytext=(0, 25), textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS[exp_key], alpha=0.8))
        
        # Add data labels to the right
        ax.text(peak_mem + 500, avg_time, 
               f'{peak_mem:.0f} MB\n{avg_time:.0f} ms',
               ha='left', va='center', fontsize=12, fontweight='bold')
    
    if all_peak_mems:
        min_mem, max_mem = min(all_peak_mems), max(all_peak_mems)
        mem_range = max_mem - min_mem if max_mem > min_mem else max_mem * 0.1
        ax.set_xlim(min_mem - mem_range * 0.2, max_mem + mem_range * 0.4) # Adjust xlim for labels

    if all_avg_times:
        min_time, max_time = min(all_avg_times), max(all_avg_times)
        time_range = max_time - min_time if max_time > min_time else max_time * 0.1
        ax.set_ylim(min_time - time_range * 0.2, max_time + time_range * 0.2)

    ax.set_xlabel('Peak Allocated Memory (MB)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Training Time per Step (ms)', fontsize=16, fontweight='bold')
    ax.set_title('Memory vs Training Time', fontsize=18, fontweight='bold')
    ax.legend(loc='upper left', fontsize=14)
    ax.grid(True, alpha=0.5, linestyle='--')
    
    plt.tight_layout(pad=1.5)
    output_path = Path(output_dir) / "performance_tradeoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_summary_table(results, output_dir):
    """Generate summary table"""
    
    # Create text summary
    summary_lines = []
    summary_lines.append("="*100)
    summary_lines.append("DETAILED MEMORY PROFILING SUMMARY")
    summary_lines.append("="*100)
    summary_lines.append("")
    
    # Header
    summary_lines.append(f"{'Method':<30} {'Peak Memory (MB)':>20} {'Avg Time (ms)':>20} {'Memory/Time':>20}")
    summary_lines.append("-"*100)
    
    # Data rows
    for exp_key in EXPERIMENT_ORDER:
        if exp_key in results:
            name = NAME_MAPPING[exp_key]
            peak_mem = results[exp_key]["peak_memory_mb"]
            avg_time = results[exp_key]["avg_time_ms"]
            ratio = peak_mem / avg_time if avg_time > 0 else 0
            
            summary_lines.append(f"{name:<30} {peak_mem:>20.2f} {avg_time:>20.2f} {ratio:>20.2f}")
    
    summary_lines.append("="*100)
    summary_lines.append("")
    
    # Detailed breakdown
    summary_lines.append("COMPONENT-LEVEL BREAKDOWN:")
    summary_lines.append("")
    
    for exp_key in EXPERIMENT_ORDER:
        if exp_key not in results:
            continue
        
        summary_lines.append(f"\n{NAME_MAPPING[exp_key]}:")
        summary_lines.append("-" * 80)
        
        breakdown = results[exp_key]["breakdown"]
        for key, value in sorted(breakdown.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True):
            if key.endswith('_mb'):
                display_name = key.replace('_mb', '').replace('_', ' ').title()
                summary_lines.append(f"  {display_name:<40} {value:>15.2f} MB")
    
    summary_lines.append("")
    summary_lines.append("="*100)
    
    # Save to file
    output_path = Path(output_dir) / "summary.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Saved: {output_path}")
    
    # Also print to console
    print("\n" + '\n'.join(summary_lines))


def main():
    parser = argparse.ArgumentParser(description="Visualize memory breakdown")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Directory containing experiment JSON files")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Loading results...")
    print(f"{'='*80}\n")
    
    # Load results
    results = load_results(args.input_dir)
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}\n")
    
    # Generate plots
    plot_memory_breakdown_comparison(results, output_path)
    plot_memory_timeline(results, output_path)
    plot_performance_comparison(results, output_path)
    generate_summary_table(results, output_path)
    
    print(f"\n{'='*80}")
    print("✅ All visualizations generated successfully!")
    print(f"✅ Output directory: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

