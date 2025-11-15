"""
Visualize Benchmark Results

This script generates comparison plots and tables from benchmark results.

Usage:
    python visualize_results.py benchmark_results.json
"""

import json
import sys
import os
from typing import Dict, List, Any
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np


def load_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_time_comparison(results: Dict[str, Any], output_dir: str):
    """Generate time comparison bar charts."""
    benchmarks = [b for b in results['benchmarks'] if 'error' not in b]
    
    # Group by shape and tile_size
    grouped = defaultdict(list)
    for b in benchmarks:
        key = (tuple(b['A_shape']), tuple(b['G_shape']), b['tile_size'])
        grouped[key].append(b)
    
    # Create one figure per (shape, tile_size) combination
    for key, group in grouped.items():
        A_shape, G_shape, tile_size = key
        
        # Sort by algorithm name for consistent ordering
        group_sorted = sorted(group, key=lambda x: x['algorithm'])
        
        algorithms = [b['algorithm'] for b in group_sorted]
        times_ms = [b['median_time_sec'] * 1000 for b in group_sorted]
        errors_ms = [b['std_time_sec'] * 1000 for b in group_sorted]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = ax.bar(range(len(algorithms)), times_ms, 
                      yerr=errors_ms, capsize=5,
                      color=colors[:len(algorithms)],
                      edgecolor='black', linewidth=1.5)
        
        # Customize chart
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'Time Comparison\nA={A_shape}, G={G_shape}, Tile={tile_size}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, time_val) in enumerate(zip(bars, times_ms)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.1f}ms',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'time_comparison_A{A_shape[1]}x{A_shape[2]}_G{G_shape[2]}_tile{tile_size}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")


def plot_speedup_comparison(results: Dict[str, Any], output_dir: str):
    """Generate speedup comparison charts."""
    benchmarks = [b for b in results['benchmarks'] if 'error' not in b]
    
    # Group by shape and tile_size
    grouped = defaultdict(list)
    for b in benchmarks:
        key = (tuple(b['A_shape']), tuple(b['G_shape']), b['tile_size'])
        grouped[key].append(b)
    
    for key, group in grouped.items():
        A_shape, G_shape, tile_size = key
        
        # Sort by time to find baseline (slowest)
        group_sorted = sorted(group, key=lambda x: x['median_time_sec'], reverse=True)
        baseline_time = group_sorted[0]['median_time_sec']
        
        algorithms = [b['algorithm'] for b in group_sorted]
        speedups = [baseline_time / b['median_time_sec'] for b in group_sorted]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#27AE60' if s >= 1.0 else '#E74C3C' for s in speedups]
        bars = ax.bar(range(len(algorithms)), speedups,
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # Add baseline line
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline')
        
        # Customize chart
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (vs slowest)', fontsize=12, fontweight='bold')
        ax.set_title(f'Speedup Comparison\nA={A_shape}, G={G_shape}, Tile={tile_size}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend()
        
        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speedup:.2f}x',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'speedup_comparison_A{A_shape[1]}x{A_shape[2]}_G{G_shape[2]}_tile{tile_size}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")


def plot_memory_comparison(results: Dict[str, Any], output_dir: str):
    """Generate memory usage comparison charts."""
    benchmarks = [b for b in results['benchmarks'] if 'error' not in b]
    
    # Group by shape and tile_size
    grouped = defaultdict(list)
    for b in benchmarks:
        key = (tuple(b['A_shape']), tuple(b['G_shape']), b['tile_size'])
        grouped[key].append(b)
    
    for key, group in grouped.items():
        A_shape, G_shape, tile_size = key
        
        # Sort by algorithm name
        group_sorted = sorted(group, key=lambda x: x['algorithm'])
        
        algorithms = [b['algorithm'] for b in group_sorted]
        peak_memory = [b['peak_memory_mb'] for b in group_sorted]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498DB', '#9B59B6', '#E67E22', '#E74C3C']
        bars = ax.bar(range(len(algorithms)), peak_memory,
                      color=colors[:len(algorithms)],
                      edgecolor='black', linewidth=1.5)
        
        # Customize chart
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
        ax.set_title(f'Memory Usage Comparison\nA={A_shape}, G={G_shape}, Tile={tile_size}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, mem in zip(bars, peak_memory):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mem:.1f}MB',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'memory_comparison_A{A_shape[1]}x{A_shape[2]}_G{G_shape[2]}_tile{tile_size}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")


def plot_tile_size_analysis(results: Dict[str, Any], output_dir: str):
    """Generate tile size analysis plots."""
    benchmarks = [b for b in results['benchmarks'] if 'error' not in b]
    
    # Group by shape and algorithm
    grouped = defaultdict(list)
    for b in benchmarks:
        key = (tuple(b['A_shape']), tuple(b['G_shape']), b['algorithm'])
        grouped[key].append(b)
    
    for key, group in grouped.items():
        A_shape, G_shape, algorithm = key
        
        # Sort by tile size
        group_sorted = sorted(group, key=lambda x: x['tile_size'])
        
        tile_sizes = [b['tile_size'] for b in group_sorted]
        times_ms = [b['median_time_sec'] * 1000 for b in group_sorted]
        
        if len(tile_sizes) < 2:
            continue  # Skip if only one tile size
        
        # Create line plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(tile_sizes, times_ms, marker='o', linewidth=2, 
               markersize=8, color='#2E86AB')
        
        # Customize chart
        ax.set_xlabel('Tile Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'Tile Size Impact - {algorithm}\nA={A_shape}, G={G_shape}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xscale('log', base=2)
        
        # Add value labels
        for tile, time_val in zip(tile_sizes, times_ms):
            ax.annotate(f'{time_val:.1f}ms',
                       xy=(tile, time_val),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filename = f'tile_analysis_{algorithm}_A{A_shape[1]}x{A_shape[2]}_G{G_shape[2]}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")


def generate_summary_table(results: Dict[str, Any], output_dir: str):
    """Generate summary table as text file."""
    benchmarks = [b for b in results['benchmarks'] if 'error' not in b]
    
    output_path = os.path.join(output_dir, 'summary_table.txt')
    
    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("BENCHMARK SUMMARY TABLE\n")
        f.write("="*100 + "\n\n")
        
        # Group by shape
        grouped = defaultdict(list)
        for b in benchmarks:
            key = (tuple(b['A_shape']), tuple(b['G_shape']))
            grouped[key].append(b)
        
        for shape_key, group in grouped.items():
            A_shape, G_shape = shape_key
            f.write(f"\nShape: A={A_shape}, G={G_shape}\n")
            f.write("-"*100 + "\n")
            
            # Group by tile size
            tile_grouped = defaultdict(list)
            for b in group:
                tile_grouped[b['tile_size']].append(b)
            
            for tile_size in sorted(tile_grouped.keys()):
                tile_group = tile_grouped[tile_size]
                f.write(f"\n  Tile Size: {tile_size}\n")
                f.write("  " + "-"*96 + "\n")
                
                # Sort by time
                tile_group_sorted = sorted(tile_group, key=lambda x: x['median_time_sec'])
                fastest_time = tile_group_sorted[0]['median_time_sec']
                
                # Header
                f.write(f"  {'Algorithm':<25} {'Time (ms)':>12} {'Speedup':>10} "
                       f"{'Memory (MB)':>12} {'Rel Error':>12}\n")
                f.write("  " + "-"*96 + "\n")
                
                # Data rows
                for b in tile_group_sorted:
                    speedup = fastest_time / b['median_time_sec']
                    f.write(f"  {b['algorithm']:<25} "
                           f"{b['median_time_sec']*1000:>12.2f} "
                           f"{speedup:>10.2f}x "
                           f"{b['peak_memory_mb']:>12.1f} "
                           f"{b['relative_error']:>12.2e}\n")
                
                f.write("\n")
        
        # Metadata
        f.write("\n" + "="*100 + "\n")
        f.write("METADATA\n")
        f.write("="*100 + "\n")
        metadata = results['metadata']
        f.write(f"Timestamp: {metadata['timestamp']}\n")
        f.write(f"Device: {metadata['device']}\n")
        f.write(f"CUDA Available: {metadata['cuda_available']}\n")
        f.write(f"Triton Available: {metadata['triton_available']}\n")
        if 'gpu_name' in metadata:
            f.write(f"GPU: {metadata['gpu_name']}\n")
            f.write(f"GPU Memory: {metadata['gpu_memory_gb']:.1f} GB\n")
    
    print(f"  ✓ Saved: summary_table.txt")


def generate_best_config_report(results: Dict[str, Any], output_dir: str):
    """Generate a report of best configurations."""
    benchmarks = [b for b in results['benchmarks'] if 'error' not in b]
    
    output_path = os.path.join(output_dir, 'best_configs.txt')
    
    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("BEST CONFIGURATION RECOMMENDATIONS\n")
        f.write("="*100 + "\n\n")
        
        # Group by shape
        grouped = defaultdict(list)
        for b in benchmarks:
            key = (tuple(b['A_shape']), tuple(b['G_shape']))
            grouped[key].append(b)
        
        for shape_key, group in grouped.items():
            A_shape, G_shape = shape_key
            
            # Find fastest overall
            fastest = min(group, key=lambda x: x['median_time_sec'])
            
            f.write(f"\nShape: A={A_shape}, G={G_shape}\n")
            f.write("-"*100 + "\n")
            f.write(f"  Best Algorithm: {fastest['algorithm']}\n")
            f.write(f"  Best Tile Size: {fastest['tile_size']}\n")
            f.write(f"  Time: {fastest['median_time_sec']*1000:.2f} ms\n")
            f.write(f"  Peak Memory: {fastest['peak_memory_mb']:.1f} MB\n")
            f.write(f"  Relative Error: {fastest['relative_error']:.2e}\n")
            
            # Compare with baseline (pytorch_input_length)
            baseline = [b for b in group if b['algorithm'] == 'pytorch_input_length' 
                       and b['tile_size'] == fastest['tile_size']]
            if baseline:
                baseline = baseline[0]
                speedup = baseline['median_time_sec'] / fastest['median_time_sec']
                f.write(f"  Speedup vs PyTorch baseline: {speedup:.2f}x\n")
            
            f.write("\n")
    
    print(f"  ✓ Saved: best_configs.txt")


def main():
    """Main visualization script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('input_file', type=str, nargs='?',
                       default='benchmark_results.json',
                       help='Input JSON file with benchmark results')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load results
    input_path = args.input_file
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.path.dirname(__file__), input_path)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading results from: {input_path}")
    results = load_results(input_path)
    
    # Create output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating visualizations in: {output_dir}\n")
    
    # Generate all visualizations
    print("Generating time comparison charts...")
    plot_time_comparison(results, output_dir)
    
    print("\nGenerating speedup comparison charts...")
    plot_speedup_comparison(results, output_dir)
    
    print("\nGenerating memory comparison charts...")
    plot_memory_comparison(results, output_dir)
    
    print("\nGenerating tile size analysis...")
    plot_tile_size_analysis(results, output_dir)
    
    print("\nGenerating summary tables...")
    generate_summary_table(results, output_dir)
    generate_best_config_report(results, output_dir)
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

