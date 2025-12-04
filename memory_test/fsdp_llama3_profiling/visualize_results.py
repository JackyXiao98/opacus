#!/usr/bin/env python3
"""
Visualize FSDP profiling results with detailed comparison plots.
Supports both LLM and DiT models.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Display names for modes
MODE_NAMES = {
    # Multi-GPU FSDP modes
    "no_dp": "Non-DP\nFSDP2",
    "ghost_fsdp": "Ghost\nFSDP",
    "flash_fsdp": "Flash\nFSDP",
    "flash_fsdp_bk": "Flash\nFSDP\n(BK)",
    "ghost_fsdp_bk": "Ghost\nFSDP\n(BK)",
    "flash_fsdp_fuse": "Flash\nFSDP\n(Fuse)",
    "flash_fsdp_fuse_bk": "Flash\nFSDP\n(Fuse+BK)",
    # Single-GPU modes
    "no_dp_single": "Non-DP\nSingle",
    "ghost": "Ghost\nSingle",
    "flash": "Flash\nSingle",
    "flash_bk": "Flash\nSingle\n(BK)",
    "ghost_bk": "Ghost\nSingle\n(BK)",
    "flash_fuse": "Flash\nSingle\n(Fuse)",
    "flash_fuse_bk": "Flash\nSingle\n(Fuse+BK)",
}

# Colors for modes
MODE_COLORS = {
    # Multi-GPU FSDP modes (darker shades)
    "no_dp": "#2980b9",           # Dark Blue
    "ghost_fsdp": "#c0392b",      # Dark Red
    "flash_fsdp": "#27ae60",      # Dark Green
    "flash_fsdp_bk": "#d68910",   # Dark Orange
    "ghost_fsdp_bk": "#8e44ad",   # Dark Purple
    "flash_fsdp_fuse": "#1abc9c", # Teal
    "flash_fsdp_fuse_bk": "#16a085", # Dark Teal
    # Single-GPU modes (lighter shades)
    "no_dp_single": "#5dade2",    # Light Blue
    "ghost": "#ec7063",           # Light Red
    "flash": "#58d68d",           # Light Green
    "flash_bk": "#f8c471",        # Light Orange
    "ghost_bk": "#bb8fce",        # Light Purple
    "flash_fuse": "#48c9b0",      # Light Teal
    "flash_fuse_bk": "#45b39d",   # Medium Teal
}

# Mode ordering for plots
MODE_ORDER = [
    "no_dp", "no_dp_single",
    "ghost_fsdp", "ghost", 
    "flash_fsdp", "flash",
    "flash_fsdp_bk", "flash_bk",
    "ghost_fsdp_bk", "ghost_bk",
    "flash_fsdp_fuse", "flash_fuse",
    "flash_fsdp_fuse_bk", "flash_fuse_bk"
]

# Markers for different modes in line charts
MODE_MARKERS = {
    "no_dp": "o",
    "no_dp_single": "o",
    "ghost_fsdp": "s",
    "ghost": "s",
    "flash_fsdp": "^",
    "flash": "^",
    "flash_fsdp_bk": "D",
    "flash_bk": "D",
    "ghost_fsdp_bk": "P",
    "ghost_bk": "P",
    "flash_fsdp_fuse": "H",
    "flash_fuse": "H",
    "flash_fsdp_fuse_bk": "*",
    "flash_fuse_bk": "*",
}

# Markers for sequence lengths (for scatter plots)
SEQ_MARKERS = {
    1024: "o",
    2048: "s",
    4096: "^",
    8192: "D",
    16384: "P",
    32768: "H",
}


def load_results(input_dir):
    """Load all experiment results"""
    input_path = Path(input_dir)
    results = {}
    
    # Find all result files
    for result_file in input_path.glob("*_result.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
            mode = data["mode"]
            seq_len = data["seq_length"]
            
            if mode not in results:
                results[mode] = {}
            results[mode][seq_len] = data
            
            print(f"✓ Loaded {mode}, seq_length={seq_len}")
    
    return results


def plot_memory_comparison(results, output_dir):
    """Plot memory comparison by sequence length"""
    seq_lengths = sorted(set(
        seq_len for mode_data in results.values() for seq_len in mode_data.keys()
    ))
    # Sort modes by defined order, only include modes present in results
    modes = [m for m in MODE_ORDER if m in results]
    
    fig, axes = plt.subplots(1, len(seq_lengths), figsize=(6 * len(seq_lengths), 6))
    
    if len(seq_lengths) == 1:
        axes = [axes]
    
    for idx, seq_len in enumerate(seq_lengths):
        ax = axes[idx]
        
        x = np.arange(len(modes))
        memories = []
        
        for mode in modes:
            if mode in results and seq_len in results[mode]:
                memories.append(results[mode][seq_len]["peak_memory_gb"])
            else:
                memories.append(0)
        
        bars = ax.bar(x, memories, color=[MODE_COLORS[mode] for mode in modes], 
                     edgecolor='black', linewidth=2, width=0.6)
        
        # Add value labels on bars
        for i, (bar, mem) in enumerate(zip(bars, memories)):
            if mem > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       f'{mem:.1f} GB', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Training Mode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Peak Memory (GB)', fontsize=14, fontweight='bold')
        ax.set_title(f'Sequence Length = {seq_len}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_NAMES[mode] for mode in modes], fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "memory_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_time_comparison(results, output_dir):
    """Plot time comparison by sequence length"""
    seq_lengths = sorted(set(
        seq_len for mode_data in results.values() for seq_len in mode_data.keys()
    ))
    # Sort modes by defined order, only include modes present in results
    modes = [m for m in MODE_ORDER if m in results]
    
    fig, axes = plt.subplots(1, len(seq_lengths), figsize=(6 * len(seq_lengths), 6))
    
    if len(seq_lengths) == 1:
        axes = [axes]
    
    for idx, seq_len in enumerate(seq_lengths):
        ax = axes[idx]
        
        x = np.arange(len(modes))
        times = []
        
        for mode in modes:
            if mode in results and seq_len in results[mode]:
                times.append(results[mode][seq_len]["avg_time_ms"])
            else:
                times.append(0)
        
        bars = ax.bar(x, times, color=[MODE_COLORS[mode] for mode in modes],
                     edgecolor='black', linewidth=2, width=0.6)
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, times)):
            if time > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                       f'{time:.0f} ms', ha='center', va='bottom',
                       fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Training Mode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Avg Time per Iteration (ms)', fontsize=14, fontweight='bold')
        ax.set_title(f'Sequence Length = {seq_len}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_NAMES[mode] for mode in modes], fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "time_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_time_vs_seq_length(results, output_dir):
    """Plot time vs sequence length as a line chart (one line per mode)"""
    # Get all sequence lengths across all modes
    seq_lengths = sorted(set(
        seq_len for mode_data in results.values() for seq_len in mode_data.keys()
    ))
    
    # Need at least 2 sequence lengths to make a meaningful line chart
    if len(seq_lengths) < 2:
        print("⚠️  Need at least 2 sequence lengths to plot time vs sequence length, skipping")
        return
    
    # Sort modes by defined order, only include modes present in results
    modes = [m for m in MODE_ORDER if m in results]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for mode in modes:
        mode_seq_lengths = []
        mode_times = []
        
        for seq_len in seq_lengths:
            if seq_len in results[mode]:
                mode_seq_lengths.append(seq_len)
                mode_times.append(results[mode][seq_len]["avg_time_ms"])
        
        if mode_seq_lengths:
            marker = MODE_MARKERS.get(mode, 'o')
            ax.plot(mode_seq_lengths, mode_times,
                   marker=marker,
                   color=MODE_COLORS[mode],
                   linewidth=2.5,
                   markersize=10,
                   label=MODE_NAMES[mode].replace('\n', ' '),
                   alpha=0.8)
    
    ax.set_xlabel('Sequence Length', fontsize=16, fontweight='bold')
    ax.set_ylabel('Avg Time per Iteration (ms)', fontsize=16, fontweight='bold')
    ax.set_title('Time vs Sequence Length', fontsize=18, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # Set x-axis to show all sequence lengths
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels(seq_lengths, fontsize=12)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "time_vs_seq_length.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_memory_vs_time_tradeoff(results, output_dir):
    """Plot memory vs time tradeoff"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort modes by defined order, only include modes present in results
    modes = [m for m in MODE_ORDER if m in results]
    
    for mode in modes:
        for seq_len, data in results[mode].items():
            memory_gb = data["peak_memory_gb"]
            time_ms = data["avg_time_ms"]
            
            marker = SEQ_MARKERS.get(seq_len, 'o')
            ax.scatter(memory_gb, time_ms, 
                      s=400,
                      color=MODE_COLORS[mode],
                      marker=marker,
                      edgecolors='black',
                      linewidth=2,
                      alpha=0.8,
                      label=f"{MODE_NAMES[mode].replace(chr(10), ' ')} (seq={seq_len})")
            
            # Add annotation
            ax.annotate(f"{MODE_NAMES[mode].replace(chr(10), ' ')}\nseq={seq_len}",
                       xy=(memory_gb, time_ms),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor=MODE_COLORS[mode], 
                                alpha=0.6,
                                edgecolor='black'))
    
    ax.set_xlabel('Peak Memory (GB)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Avg Time per Iteration (ms)', fontsize=16, fontweight='bold')
    ax.set_title('Memory vs Time Tradeoff', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for mode in modes:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=MODE_COLORS[mode], 
                  markersize=12, markeredgecolor='black', markeredgewidth=2,
                  label=MODE_NAMES[mode].replace('\n', ' '))
        )
    
    ax.legend(handles=legend_elements, loc='best', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "memory_vs_time_tradeoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_overhead_analysis(results, output_dir):
    """Plot overhead analysis (DP vs non-DP) for both FSDP and Single-GPU modes"""
    
    # Determine which baselines are available
    has_fsdp_baseline = "no_dp" in results
    has_single_baseline = "no_dp_single" in results
    
    if not has_fsdp_baseline and not has_single_baseline:
        print("⚠️  No baseline results found, skipping overhead analysis")
        return
    
    # Create subplots
    num_plots = sum([has_fsdp_baseline, has_single_baseline])
    fig, axes = plt.subplots(num_plots, 2, figsize=(16, 6 * num_plots))
    
    if num_plots == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    
    # Helper function to plot overhead for a specific baseline
    def plot_baseline_overhead(baseline_mode, dp_mode_list, ax_mem, ax_time, title_prefix):
        seq_lengths = sorted(results[baseline_mode].keys())
        x = np.arange(len(seq_lengths))
        width = 0.8 / len(dp_mode_list)
        
        # Memory overhead
        for idx, mode in enumerate(dp_mode_list):
            memory_overheads = []
            for seq_len in seq_lengths:
                if seq_len in results[baseline_mode] and seq_len in results[mode]:
                    baseline = results[baseline_mode][seq_len]["peak_memory_gb"]
                    dp_mem = results[mode][seq_len]["peak_memory_gb"]
                    overhead = dp_mem - baseline
                    memory_overheads.append(overhead)
                else:
                    memory_overheads.append(0)
            
            offset = width * (idx - len(dp_mode_list) / 2 + 0.5)
            bars = ax_mem.bar(x + offset, memory_overheads, width,
                          label=MODE_NAMES[mode].replace('\n', ' '),
                          color=MODE_COLORS[mode],
                          edgecolor='black',
                          linewidth=1.5)
            
            # Add value labels
            for bar, overhead in zip(bars, memory_overheads):
                if overhead > 0:
                    ax_mem.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f'{overhead:.1f}', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
        
        ax_mem.set_xlabel('Sequence Length', fontsize=14, fontweight='bold')
        ax_mem.set_ylabel('Memory Overhead (GB)', fontsize=14, fontweight='bold')
        ax_mem.set_title(f'{title_prefix} - Memory Overhead', fontsize=16, fontweight='bold')
        ax_mem.set_xticks(x)
        ax_mem.set_xticklabels(seq_lengths, fontsize=12)
        ax_mem.legend(fontsize=10, loc='best')
        ax_mem.grid(axis='y', alpha=0.3, linestyle='--')
        ax_mem.axhline(y=0, color='black', linewidth=1)
        
        # Time overhead (percentage)
        for idx, mode in enumerate(dp_mode_list):
            time_overheads = []
            for seq_len in seq_lengths:
                if seq_len in results[baseline_mode] and seq_len in results[mode]:
                    baseline = results[baseline_mode][seq_len]["avg_time_ms"]
                    dp_time = results[mode][seq_len]["avg_time_ms"]
                    overhead_pct = ((dp_time - baseline) / baseline) * 100 if baseline > 0 else 0
                    time_overheads.append(overhead_pct)
                else:
                    time_overheads.append(0)
            
            offset = width * (idx - len(dp_mode_list) / 2 + 0.5)
            bars = ax_time.bar(x + offset, time_overheads, width,
                          label=MODE_NAMES[mode].replace('\n', ' '),
                          color=MODE_COLORS[mode],
                          edgecolor='black',
                          linewidth=1.5)
            
            # Add value labels
            for bar, overhead in zip(bars, time_overheads):
                if abs(overhead) > 0.1:
                    ax_time.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f'{overhead:.1f}%', ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
        
        ax_time.set_xlabel('Sequence Length', fontsize=14, fontweight='bold')
        ax_time.set_ylabel('Time Overhead (%)', fontsize=14, fontweight='bold')
        ax_time.set_title(f'{title_prefix} - Time Overhead', fontsize=16, fontweight='bold')
        ax_time.set_xticks(x)
        ax_time.set_xticklabels(seq_lengths, fontsize=12)
        ax_time.legend(fontsize=10, loc='best')
        ax_time.grid(axis='y', alpha=0.3, linestyle='--')
        ax_time.axhline(y=0, color='black', linewidth=1)
    
    # Plot FSDP overhead
    if has_fsdp_baseline:
        fsdp_dp_modes = [m for m in ["ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk"] 
                         if m in results]
        if fsdp_dp_modes:
            plot_baseline_overhead("no_dp", fsdp_dp_modes, 
                                 axes[plot_idx, 0], axes[plot_idx, 1],
                                 "FSDP Modes (vs no_dp)")
            plot_idx += 1
    
    # Plot Single-GPU overhead
    if has_single_baseline:
        single_dp_modes = [m for m in ["ghost", "flash", "flash_bk", "ghost_bk", "flash_fuse", "flash_fuse_bk"] 
                          if m in results]
        if single_dp_modes:
            plot_baseline_overhead("no_dp_single", single_dp_modes,
                                 axes[plot_idx, 0], axes[plot_idx, 1],
                                 "Single-GPU Modes (vs no_dp_single)")
            plot_idx += 1
    
    plt.tight_layout()
    output_path = Path(output_dir) / "overhead_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_summary_table(results, output_dir):
    """Generate summary table"""
    summary_lines = []
    summary_lines.append("=" * 110)
    summary_lines.append("PROFILING SUMMARY (FSDP + Single-GPU)")
    summary_lines.append("=" * 110)
    summary_lines.append("")
    
    # Get all modes and sequence lengths
    modes = [m for m in MODE_ORDER if m in results]
    seq_lengths = sorted(set(
        seq_len for mode_data in results.values() for seq_len in mode_data.keys()
    ))
    
    # Header
    summary_lines.append(f"{'Mode':<25} {'Seq Length':>12} {'Peak Mem (GB)':>15} {'Avg Time (ms)':>15} {'GPUs':>8} {'Batch/GPU':>12}")
    summary_lines.append("-" * 110)
    
    # Data rows
    for mode in modes:
        for seq_len in seq_lengths:
            if seq_len in results[mode]:
                data = results[mode][seq_len]
                mode_name = MODE_NAMES[mode].replace('\n', ' ')
                num_gpus = data.get('num_gpus', 1)
                batch_per_gpu = data.get('batch_size', data['total_batch_size'] // num_gpus)
                summary_lines.append(
                    f"{mode_name:<25} {seq_len:>12} {data['peak_memory_gb']:>15.2f} "
                    f"{data['avg_time_ms']:>15.2f} {num_gpus:>8} {batch_per_gpu:>12}"
                )
        summary_lines.append("-" * 110)
    
    summary_lines.append("")
    
    # Overhead analysis for FSDP modes
    if "no_dp" in results:
        summary_lines.append("OVERHEAD ANALYSIS - FSDP MODES (vs no_dp baseline):")
        summary_lines.append("")
        
        fsdp_dp_modes = [m for m in modes if m in ["ghost_fsdp", "flash_fsdp", "flash_fsdp_bk", "ghost_fsdp_bk", "flash_fsdp_fuse", "flash_fsdp_fuse_bk"]]
        
        if fsdp_dp_modes:
            summary_lines.append(f"{'Mode':<25} {'Seq Length':>12} {'Mem Overhead (GB)':>20} {'Time Overhead (%)':>20}")
            summary_lines.append("-" * 110)
            
            for mode in fsdp_dp_modes:
                for seq_len in seq_lengths:
                    if seq_len in results["no_dp"] and seq_len in results[mode]:
                        baseline_mem = results["no_dp"][seq_len]["peak_memory_gb"]
                        dp_mem = results[mode][seq_len]["peak_memory_gb"]
                        mem_overhead = dp_mem - baseline_mem
                        
                        baseline_time = results["no_dp"][seq_len]["avg_time_ms"]
                        dp_time = results[mode][seq_len]["avg_time_ms"]
                        time_overhead = ((dp_time - baseline_time) / baseline_time) * 100 if baseline_time > 0 else 0
                        
                        mode_name = MODE_NAMES[mode].replace('\n', ' ')
                        summary_lines.append(
                            f"{mode_name:<25} {seq_len:>12} {mem_overhead:>20.2f} {time_overhead:>20.2f}"
                        )
                summary_lines.append("-" * 110)
            
            summary_lines.append("")
    
    # Overhead analysis for Single-GPU modes
    if "no_dp_single" in results:
        summary_lines.append("OVERHEAD ANALYSIS - SINGLE-GPU MODES (vs no_dp_single baseline):")
        summary_lines.append("")
        
        single_dp_modes = [m for m in modes if m in ["ghost", "flash", "flash_bk", "ghost_bk", "flash_fuse", "flash_fuse_bk"]]
        
        if single_dp_modes:
            summary_lines.append(f"{'Mode':<25} {'Seq Length':>12} {'Mem Overhead (GB)':>20} {'Time Overhead (%)':>20}")
            summary_lines.append("-" * 110)
            
            for mode in single_dp_modes:
                for seq_len in seq_lengths:
                    if seq_len in results["no_dp_single"] and seq_len in results[mode]:
                        baseline_mem = results["no_dp_single"][seq_len]["peak_memory_gb"]
                        dp_mem = results[mode][seq_len]["peak_memory_gb"]
                        mem_overhead = dp_mem - baseline_mem
                        
                        baseline_time = results["no_dp_single"][seq_len]["avg_time_ms"]
                        dp_time = results[mode][seq_len]["avg_time_ms"]
                        time_overhead = ((dp_time - baseline_time) / baseline_time) * 100 if baseline_time > 0 else 0
                        
                        mode_name = MODE_NAMES[mode].replace('\n', ' ')
                        summary_lines.append(
                            f"{mode_name:<25} {seq_len:>12} {mem_overhead:>20.2f} {time_overhead:>20.2f}"
                        )
                summary_lines.append("-" * 110)
    
    summary_lines.append("")
    summary_lines.append("=" * 110)
    
    # Save to file
    output_path = Path(output_dir) / "summary.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Saved: {output_path}")
    
    # Also print to console
    print("\n" + '\n'.join(summary_lines))


def main():
    parser = argparse.ArgumentParser(description="Visualize FSDP profiling results")
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
    plot_memory_comparison(results, output_path)
    plot_time_comparison(results, output_path)
    plot_time_vs_seq_length(results, output_path)
    plot_memory_vs_time_tradeoff(results, output_path)
    plot_overhead_analysis(results, output_path)
    
    generate_summary_table(results, output_path)
    
    print(f"\n{'='*80}")
    print("✅ All visualizations generated successfully!")
    print(f"✅ Output directory: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
