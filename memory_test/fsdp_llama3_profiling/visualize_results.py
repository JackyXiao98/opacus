#!/usr/bin/env python3
"""
Visualize FSDP Llama3 profiling results with detailed comparison plots.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Display names for modes
MODE_NAMES = {
    "no_dp": "Non-DP\nFSDP2",
    "ghost_fsdp": "Ghost\nClipping",
    "flash_fsdp": "Flash\nClipping",
}

# Colors for modes
MODE_COLORS = {
    "no_dp": "#3498db",      # Blue
    "ghost_fsdp": "#e74c3c",  # Red
    "flash_fsdp": "#2ecc71",  # Green
}

# Markers for sequence lengths
SEQ_MARKERS = {
    1024: "o",
    2048: "s",
    4096: "^",
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
    modes = sorted(results.keys(), key=lambda x: ["no_dp", "ghost_fsdp", "flash_fsdp"].index(x))
    
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
    modes = sorted(results.keys(), key=lambda x: ["no_dp", "ghost_fsdp", "flash_fsdp"].index(x))
    
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


def plot_memory_vs_time_tradeoff(results, output_dir):
    """Plot memory vs time tradeoff"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    modes = sorted(results.keys(), key=lambda x: ["no_dp", "ghost_fsdp", "flash_fsdp"].index(x))
    
    for mode in modes:
        for seq_len, data in results[mode].items():
            memory_gb = data["peak_memory_gb"]
            time_ms = data["avg_time_ms"]
            
            ax.scatter(memory_gb, time_ms, 
                      s=400,
                      color=MODE_COLORS[mode],
                      marker=SEQ_MARKERS[seq_len],
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
    """Plot overhead analysis (DP vs non-DP)"""
    if "no_dp" not in results:
        print("⚠️  No baseline (no_dp) results found, skipping overhead analysis")
        return
    
    seq_lengths = sorted(results["no_dp"].keys())
    dp_modes = [m for m in results.keys() if m != "no_dp"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Memory overhead
    x = np.arange(len(seq_lengths))
    width = 0.35
    
    for idx, mode in enumerate(dp_modes):
        memory_overheads = []
        for seq_len in seq_lengths:
            if seq_len in results["no_dp"] and seq_len in results[mode]:
                baseline = results["no_dp"][seq_len]["peak_memory_gb"]
                dp_mem = results[mode][seq_len]["peak_memory_gb"]
                overhead = dp_mem - baseline
                memory_overheads.append(overhead)
            else:
                memory_overheads.append(0)
        
        offset = width * (idx - len(dp_modes) / 2 + 0.5)
        bars = ax1.bar(x + offset, memory_overheads, width,
                      label=MODE_NAMES[mode].replace('\n', ' '),
                      color=MODE_COLORS[mode],
                      edgecolor='black',
                      linewidth=1.5)
        
        # Add value labels
        for bar, overhead in zip(bars, memory_overheads):
            if overhead > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{overhead:.1f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Sequence Length', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Memory Overhead (GB)', fontsize=14, fontweight='bold')
    ax1.set_title('Memory Overhead vs Non-DP Baseline', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_lengths, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='black', linewidth=1)
    
    # Time overhead (percentage)
    for idx, mode in enumerate(dp_modes):
        time_overheads = []
        for seq_len in seq_lengths:
            if seq_len in results["no_dp"] and seq_len in results[mode]:
                baseline = results["no_dp"][seq_len]["avg_time_ms"]
                dp_time = results[mode][seq_len]["avg_time_ms"]
                overhead_pct = ((dp_time - baseline) / baseline) * 100 if baseline > 0 else 0
                time_overheads.append(overhead_pct)
            else:
                time_overheads.append(0)
        
        offset = width * (idx - len(dp_modes) / 2 + 0.5)
        bars = ax2.bar(x + offset, time_overheads, width,
                      label=MODE_NAMES[mode].replace('\n', ' '),
                      color=MODE_COLORS[mode],
                      edgecolor='black',
                      linewidth=1.5)
        
        # Add value labels
        for bar, overhead in zip(bars, time_overheads):
            if abs(overhead) > 0.1:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{overhead:.1f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Sequence Length', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time Overhead (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Time Overhead vs Non-DP Baseline', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_lengths, fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linewidth=1)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "overhead_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_summary_table(results, output_dir):
    """Generate summary table"""
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("FSDP LLAMA3 PROFILING SUMMARY")
    summary_lines.append("=" * 100)
    summary_lines.append("")
    
    # Get all modes and sequence lengths
    modes = sorted(results.keys(), key=lambda x: ["no_dp", "ghost_fsdp", "flash_fsdp"].index(x))
    seq_lengths = sorted(set(
        seq_len for mode_data in results.values() for seq_len in mode_data.keys()
    ))
    
    # Header
    summary_lines.append(f"{'Mode':<20} {'Seq Length':>12} {'Peak Mem (GB)':>15} {'Avg Time (ms)':>15} {'Batch Size':>12}")
    summary_lines.append("-" * 100)
    
    # Data rows
    for mode in modes:
        for seq_len in seq_lengths:
            if seq_len in results[mode]:
                data = results[mode][seq_len]
                mode_name = MODE_NAMES[mode].replace('\n', ' ')
                summary_lines.append(
                    f"{mode_name:<20} {seq_len:>12} {data['peak_memory_gb']:>15.2f} "
                    f"{data['avg_time_ms']:>15.2f} {data['total_batch_size']:>12}"
                )
        summary_lines.append("-" * 100)
    
    summary_lines.append("")
    
    # Overhead analysis
    if "no_dp" in results:
        summary_lines.append("OVERHEAD ANALYSIS (vs Non-DP Baseline):")
        summary_lines.append("")
        
        dp_modes = [m for m in modes if m != "no_dp"]
        
        summary_lines.append(f"{'Mode':<20} {'Seq Length':>12} {'Mem Overhead (GB)':>20} {'Time Overhead (%)':>20}")
        summary_lines.append("-" * 100)
        
        for mode in dp_modes:
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
                        f"{mode_name:<20} {seq_len:>12} {mem_overhead:>20.2f} {time_overhead:>20.2f}"
                    )
            summary_lines.append("-" * 100)
    
    summary_lines.append("")
    summary_lines.append("=" * 100)
    
    # Save to file
    output_path = Path(output_dir) / "summary.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Saved: {output_path}")
    
    # Also print to console
    print("\n" + '\n'.join(summary_lines))


def main():
    parser = argparse.ArgumentParser(description="Visualize FSDP Llama3 profiling results")
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
    plot_memory_vs_time_tradeoff(results, output_path)
    plot_overhead_analysis(results, output_path)
    generate_summary_table(results, output_path)
    
    print(f"\n{'='*80}")
    print("✅ All visualizations generated successfully!")
    print(f"✅ Output directory: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

