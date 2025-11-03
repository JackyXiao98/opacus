#!/usr/bin/env python3
"""
Detailed memory profiling for per-sample gradient norm computation methods.

This script provides line-by-line memory analysis for:
1. Ghost Clipping method (O(T²) memory complexity)
2. Triton FC-PathB method (O(T) memory complexity)

Focus on analyzing memory usage at each step of the computation.
"""

import gc
import time
import tracemalloc
from typing import Dict, List, Tuple
import psutil
import os
import sys

import torch
import torch.nn as nn
from opt_einsum import contract

# Try to import triton functions, but continue without them if not available
try:
    from triton_version.triton_flash_clipping_linear import compute_linear_norm_sample_triton, _triton_frobenius_inner_over_T, _triton_sum_over_time_norm_squared
    TRITON_AVAILABLE = True
    print("✅ Triton functions imported successfully")
except ImportError as e:
    print(f"⚠️  Triton not available: {e}")
    print("📊 Will only analyze Ghost Clipping method")
    TRITON_AVAILABLE = False

# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

def print_header(title: str, width: int = 100, style: str = "double"):
    """Print a beautiful header with different styles"""
    if style == "double":
        top_line = "╔" + "═" * (width - 2) + "╗"
        bottom_line = "╚" + "═" * (width - 2) + "╝"
        border_char = "║"
    elif style == "single":
        top_line = "┌" + "─" * (width - 2) + "┐"
        bottom_line = "└" + "─" * (width - 2) + "┘"
        border_char = "│"
    else:  # simple
        top_line = "+" + "=" * (width - 2) + "+"
        bottom_line = "+" + "=" * (width - 2) + "+"
        border_char = "|"
    
    # Calculate padding for centered text
    title_len = len(title)
    padding = (width - 4 - title_len) // 2
    remaining = width - 4 - title_len - padding
    
    print(f"{Colors.BOLD}{Colors.OKBLUE}{top_line}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{border_char}{Colors.ENDC} {' ' * padding}{Colors.BOLD}{Colors.HEADER}{title}{Colors.ENDC}{' ' * remaining} {Colors.BOLD}{Colors.OKBLUE}{border_char}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{bottom_line}{Colors.ENDC}")

def print_section(title: str, width: int = 80):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'─' * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}📋 {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'─' * width}{Colors.ENDC}")

def format_memory(memory_mb: float) -> str:
    """Format memory value with appropriate units and colors"""
    if memory_mb >= 1024:
        value = memory_mb / 1024
        unit = "GB"
        color = Colors.FAIL if value > 10 else Colors.WARNING if value > 1 else Colors.OKGREEN
    else:
        value = memory_mb
        unit = "MB"
        color = Colors.WARNING if value > 1000 else Colors.OKGREEN if value > 100 else Colors.OKCYAN
    
    return f"{color}{value:8.2f} {unit}{Colors.ENDC}"

def format_delta(delta: float) -> str:
    """Format memory delta with colors"""
    if abs(delta) < 0.01:
        return f"{Colors.OKCYAN}    ~0.00{Colors.ENDC}"
    elif delta > 0:
        return f"{Colors.WARNING}+{delta:7.2f}{Colors.ENDC}"
    else:
        return f"{Colors.OKGREEN}{delta:8.2f}{Colors.ENDC}"

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory_usage() -> Tuple[float, float]:
    """Get GPU memory usage in MB (allocated, cached)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        cached = torch.cuda.memory_reserved() / 1024 / 1024
        return allocated, cached
    return 0.0, 0.0

class DetailedMemoryProfiler:
    """Detailed memory profiler that tracks memory at each step"""
    
    def __init__(self, name: str):
        self.name = name
        self.memory_snapshots = []
        self.gpu_snapshots = []
        self.step_names = []
        
    def snapshot(self, step_name: str):
        """Take a memory snapshot at a specific step"""
        cpu_mem = get_memory_usage()
        gpu_alloc, gpu_cached = get_gpu_memory_usage()
        
        self.memory_snapshots.append(cpu_mem)
        self.gpu_snapshots.append(gpu_alloc)
        self.step_names.append(step_name)
        
        # Calculate deltas
        if len(self.memory_snapshots) > 1:
            cpu_delta = cpu_mem - self.memory_snapshots[-2]
            gpu_delta = gpu_alloc - self.gpu_snapshots[-2]
        else:
            cpu_delta = 0.0
            gpu_delta = 0.0
        
        print(f"  {Colors.OKCYAN}▶{Colors.ENDC} {step_name:<45} CPU: {format_memory(cpu_mem)} GPU: {format_memory(gpu_alloc)} Δ: {format_delta(cpu_delta)} / {format_delta(gpu_delta)}")
        
    def print_summary(self):
        """Print detailed memory usage summary"""
        print_section(f"MEMORY PROFILE SUMMARY: {self.name}")
        
        if len(self.memory_snapshots) < 2:
            print(f"{Colors.WARNING}⚠️  Not enough snapshots for analysis{Colors.ENDC}")
            return
        
        # Table header
        print(f"\n{Colors.BOLD}{'Step':<45} {'CPU Memory':<15} {'GPU Memory':<15} {'CPU Δ':<12} {'GPU Δ':<12}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'─' * 45} {'─' * 15} {'─' * 15} {'─' * 12} {'─' * 12}{Colors.ENDC}")
        
        for i, (step, cpu_mem, gpu_mem) in enumerate(zip(self.step_names, self.memory_snapshots, self.gpu_snapshots)):
            if i == 0:
                cpu_delta = 0.0
                gpu_delta = 0.0
            else:
                cpu_delta = cpu_mem - self.memory_snapshots[i-1]
                gpu_delta = gpu_mem - self.gpu_snapshots[i-1]
            
            step_display = step[:42] + "..." if len(step) > 45 else step
            print(f"{step_display:<45} {format_memory(cpu_mem):<24} {format_memory(gpu_mem):<24} {format_delta(cpu_delta):<21} {format_delta(gpu_delta):<21}")
        
        # Summary statistics
        total_cpu_delta = self.memory_snapshots[-1] - self.memory_snapshots[0]
        total_gpu_delta = self.gpu_snapshots[-1] - self.gpu_snapshots[0]
        peak_cpu = max(self.memory_snapshots)
        peak_gpu = max(self.gpu_snapshots)
        
        print(f"{Colors.BOLD}{'─' * 45} {'─' * 15} {'─' * 15} {'─' * 12} {'─' * 12}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'TOTAL CHANGE':<45}{Colors.ENDC} {'':<15} {'':<15} {format_delta(total_cpu_delta):<21} {format_delta(total_gpu_delta):<21}")
        print(f"{Colors.BOLD}{'PEAK USAGE':<45}{Colors.ENDC} {format_memory(peak_cpu):<24} {format_memory(peak_gpu):<24}")
        
        # Memory efficiency indicators
        if peak_cpu > 1000:
            print(f"\n{Colors.WARNING}⚠️  High CPU memory usage detected: {peak_cpu:.2f} MB{Colors.ENDC}")
        if peak_gpu > 1000:
            print(f"\n{Colors.WARNING}⚠️  High GPU memory usage detected: {peak_gpu:.2f} MB{Colors.ENDC}")

def create_test_data(B: int, T: int, d: int, p: int, device: str = 'cuda', dtype: torch.dtype = torch.float32):
    """Create test data with specified dimensions"""
    print_section("DATA CREATION")
    print(f"{Colors.BOLD}📊 Creating test data:{Colors.ENDC} B={B}, T={T}, d={d}, p={p}")
    print(f"{Colors.BOLD}🖥️  Device:{Colors.ENDC} {device}, {Colors.BOLD}📏 dtype:{Colors.ENDC} {dtype}")
    
    profiler = DetailedMemoryProfiler("Data Creation")
    profiler.snapshot("🚀 Before data creation")
    
    # Create activations [B, T, d]
    activations = torch.randn(B, T, d, device=device, dtype=dtype)
    profiler.snapshot(f"📈 Activations created [{B}, {T}, {d}]")
    
    # Create gradients [B, T, p] 
    gradients = torch.randn(B, T, p, device=device, dtype=dtype)
    profiler.snapshot(f"📉 Gradients created [{B}, {T}, {p}]")
    
    # Create a dummy linear layer
    layer = nn.Linear(d, p, bias=True).to(device=device, dtype=dtype)
    profiler.snapshot("🔗 Linear layer created")
    
    # Calculate theoretical memory usage
    activation_memory = B * T * d * 4 / 1024 / 1024  # float32 = 4 bytes
    gradient_memory = B * T * p * 4 / 1024 / 1024
    layer_weight_memory = d * p * 4 / 1024 / 1024
    layer_bias_memory = p * 4 / 1024 / 1024
    
    print(f"\n{Colors.BOLD}📊 Theoretical Memory Usage:{Colors.ENDC}")
    print(f"  📈 Activations [{B}, {T}, {d}]: {format_memory(activation_memory)}")
    print(f"  📉 Gradients [{B}, {T}, {p}]: {format_memory(gradient_memory)}")
    print(f"  ⚖️  Layer weight [{d}, {p}]: {format_memory(layer_weight_memory)}")
    print(f"  ➕ Layer bias [{p}]: {format_memory(layer_bias_memory)}")
    print(f"  {Colors.BOLD}📊 Total input data: {format_memory(activation_memory + gradient_memory + layer_weight_memory + layer_bias_memory)}{Colors.ENDC}")
    
    profiler.print_summary()
    
    return activations, gradients, layer

def analyze_ghost_clipping_method(layer: nn.Linear, activations: torch.Tensor, backprops: torch.Tensor):
    """Detailed analysis of Ghost Clipping method memory usage"""
    print_header("ANALYZING GHOST CLIPPING METHOD (O(T²) complexity)", style="double")
    
    B, T, d = activations.shape
    _, _, p = backprops.shape
    
    profiler = DetailedMemoryProfiler("Ghost Clipping")
    
    # Clean up before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    profiler.snapshot("🚀 Start of computation")
    
    # Convert activations to backprops dtype
    activations = activations.to(backprops.dtype)
    profiler.snapshot("🔄 After dtype conversion")
    
    ret = {}
    
    if layer.weight.requires_grad:
        print(f"\n{Colors.BOLD}{Colors.OKGREEN}🔍 Analyzing weight gradient norm computation...{Colors.ENDC}")
        
        # Step 1: Create ggT matrix [B, T, T] - This is the memory killer!
        print(f"  {Colors.WARNING}⚠️  Creating ggT matrix [{B}, {T}, {T}]...{Colors.ENDC}")
        theoretical_ggt_memory = B * T * T * 4 / 1024 / 1024
        print(f"  📊 Theoretical ggT memory: {format_memory(theoretical_ggt_memory)}")
        
        ggT = torch.einsum("nik,njk->nij", backprops, backprops)  # [B, T, T]
        profiler.snapshot(f"💥 ggT matrix created [{B}, {T}, {T}]")
        
        # Step 2: Create aaT matrix [B, T, T] - Another memory killer!
        print(f"  {Colors.WARNING}⚠️  Creating aaT matrix [{B}, {T}, {T}]...{Colors.ENDC}")
        theoretical_aat_memory = B * T * T * 4 / 1024 / 1024
        print(f"  📊 Theoretical aaT memory: {format_memory(theoretical_aat_memory)}")
        
        aaT = torch.einsum("nik,njk->nij", activations, activations)  # [B, T, T]
        profiler.snapshot(f"💥 aaT matrix created [{B}, {T}, {T}]")
        
        # Step 3: Element-wise multiplication and sum
        print(f"  {Colors.OKCYAN}🔢 Computing element-wise product and sum...{Colors.ENDC}")
        ga = torch.einsum("n...i,n...i->n", ggT, aaT).clamp(min=0)  # [B]
        profiler.snapshot("✨ Element-wise product and sum completed")
        
        # Step 4: Square root
        ret[layer.weight] = torch.sqrt(ga)
        profiler.snapshot("√ Square root for weight completed")
        
        # Clean up intermediate tensors
        del ggT, aaT, ga
        profiler.snapshot("🧹 Cleanup of ggT, aaT completed")
    
    if layer.bias is not None and layer.bias.requires_grad:
        print(f"\n{Colors.BOLD}{Colors.OKGREEN}🔍 Analyzing bias gradient norm computation...{Colors.ENDC}")
        
        # For bias, we need ggT again (if not already computed)
        print(f"  {Colors.WARNING}⚠️  Creating ggT matrix for bias [{B}, {T}, {T}]...{Colors.ENDC}")
        ggT = torch.einsum("nik,njk->nij", backprops, backprops)  # [B, T, T]
        profiler.snapshot(f"💥 ggT matrix for bias created [{B}, {T}, {T}]")
        
        # Sum over T dimensions
        bias_norm = torch.einsum("n...i->n", ggT).clamp(min=0)  # [B]
        profiler.snapshot("📊 Sum for bias completed")
        
        ret[layer.bias] = torch.sqrt(bias_norm)
        profiler.snapshot("√ Square root for bias completed")
        
        # Clean up
        del ggT, bias_norm
        profiler.snapshot("🧹 Cleanup for bias completed")
    
    profiler.print_summary()
    
    # Calculate total theoretical memory for Gram matrices
    total_gram_memory = B * T * T * 4 / 1024 / 1024 * 2  # Two T×T matrices
    print(f"\n{Colors.BOLD}{Colors.WARNING}💾 Total theoretical Gram matrix memory: {format_memory(total_gram_memory)}{Colors.ENDC}")
    
    return ret

def analyze_triton_method(layer: nn.Linear, activations: torch.Tensor, backprops: torch.Tensor, tile_size: int = 64):
    """Detailed analysis of Triton method memory usage"""
    print_header(f"ANALYZING TRITON METHOD (O(T) complexity, tile_size={tile_size})", style="double")
    
    B, T, d = activations.shape
    _, _, p = backprops.shape
    
    profiler = DetailedMemoryProfiler("Triton Method")
    
    # Clean up before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    profiler.snapshot("🚀 Start of computation")
    
    A = activations
    ret = {}
    
    if layer.weight.requires_grad:
        print(f"\n{Colors.BOLD}{Colors.OKGREEN}🔍 Analyzing Triton weight gradient norm computation...{Colors.ENDC}")
        print(f"  📊 Input shapes: A={A.shape}, backprops={backprops.shape}")
        print(f"  🔧 Tile size: {tile_size}")
        
        # Calculate theoretical tile memory
        theoretical_tile_memory = B * tile_size * tile_size * 4 / 1024 / 1024 * 2  # Two tile×tile matrices
        print(f"  💾 Theoretical tile memory per iteration: {format_memory(theoretical_tile_memory)}")
        
        # Call the Triton function with detailed profiling
        profiler.snapshot("🔄 Before _triton_frobenius_inner_over_T")
        
        # We'll manually trace through the Triton function
        print(f"  {Colors.OKCYAN}🔍 Tracing through _triton_frobenius_inner_over_T...{Colors.ENDC}")
        ga = analyze_triton_frobenius_inner(A, backprops, tile_size, profiler)
        
        profiler.snapshot("✅ After _triton_frobenius_inner_over_T")
        
        # Square root and clamp
        ret[layer.weight] = torch.sqrt(ga.clamp_min(0.0))
        profiler.snapshot("√ Square root and clamp for weight")
        
        del ga
        profiler.snapshot("🧹 After cleanup ga")
    
    if (layer.bias is not None) and layer.bias.requires_grad:
        print(f"\n{Colors.BOLD}{Colors.OKGREEN}🔍 Analyzing Triton bias gradient norm computation...{Colors.ENDC}")
        
        profiler.snapshot("🔄 Before _triton_sum_over_time_norm_squared")
        
        # For bias computation - this is simpler
        gg = analyze_triton_sum_over_time(backprops, profiler)
        
        profiler.snapshot("✅ After _triton_sum_over_time_norm_squared")
        
        ret[layer.bias] = torch.sqrt(gg.clamp_min(0.0))
        profiler.snapshot("√ Square root and clamp for bias")
        
        del gg
        profiler.snapshot("🧹 After cleanup gg")
    
    profiler.print_summary()
    
    return ret

def analyze_triton_frobenius_inner(A: torch.Tensor, G: torch.Tensor, tile_size: int, profiler: DetailedMemoryProfiler):
    """Manually trace through the Triton Frobenius inner product computation"""
    B, T, d_a = A.shape
    _, _, d_g = G.shape
    
    print(f"    {Colors.OKCYAN}🔄 Converting to accumulation dtype...{Colors.ENDC}")
    dtype_acc = torch.float32
    A = A.to(dtype_acc)
    G = G.to(dtype_acc)
    profiler.snapshot("🔄 After dtype conversion in Triton")
    
    print(f"    {Colors.OKCYAN}🆕 Initializing result tensor...{Colors.ENDC}")
    ga = torch.zeros(B, dtype=dtype_acc, device=A.device)
    profiler.snapshot("🆕 After ga initialization")
    
    num_tiles = (T + tile_size - 1) // tile_size
    print(f"    {Colors.BOLD}🔢 Number of tiles: {num_tiles}{Colors.ENDC}")
    
    for p in range(min(3, num_tiles)):  # Only trace first 3 tiles to avoid too much output
        print(f"    {Colors.BOLD}{Colors.OKBLUE}🔲 Processing tile {p+1}/{num_tiles}...{Colors.ENDC}")
        
        ps = p * tile_size
        pe = min((p + 1) * tile_size, T)
        
        # Extract tiles
        A_p = A[:, ps:pe, :]
        G_p = G[:, ps:pe, :]
        profiler.snapshot(f"📦 Extracted tile {p} [{B}, {pe-ps}, {d_a}]")
        
        # Diagonal block computation
        print(f"      {Colors.OKCYAN}⚡ Computing diagonal block ({p}, {p})...{Colors.ENDC}")
        Sg_pp = torch.bmm(G_p, G_p.transpose(-2, -1))  # [B, tau_p, tau_p]
        profiler.snapshot(f"🔢 After Sg_pp [{B}, {pe-ps}, {pe-ps}]")
        
        Sa_pp = torch.bmm(A_p, A_p.transpose(-2, -1))  # [B, tau_p, tau_p]
        profiler.snapshot(f"🔢 After Sa_pp [{B}, {pe-ps}, {pe-ps}]")
        
        ga += torch.sum(Sg_pp * Sa_pp, dim=(1, 2))
        profiler.snapshot(f"➕ After diagonal block accumulation")
        
        # Off-diagonal blocks
        for q in range(p):
            print(f"      {Colors.WARNING}🔀 Computing off-diagonal block ({p}, {q})...{Colors.ENDC}")
            qs = q * tile_size
            qe = min((q + 1) * tile_size, T)
            A_q = A[:, qs:qe, :]
            G_q = G[:, qs:qe, :]
            profiler.snapshot(f"📦 Extracted tile {q} for off-diagonal")
            
            Sg_pq = torch.bmm(G_p, G_q.transpose(-2, -1))
            Sa_pq = torch.bmm(A_p, A_q.transpose(-2, -1))
            profiler.snapshot(f"🔢 After off-diagonal bmm operations")
            
            ga += 2.0 * torch.sum(Sg_pq * Sa_pq, dim=(1, 2))
            profiler.snapshot(f"➕ After off-diagonal accumulation")
            
            # Clean up off-diagonal temporaries
            del A_q, G_q, Sg_pq, Sa_pq
        
        # Clean up diagonal temporaries
        del A_p, G_p, Sg_pp, Sa_pp
        profiler.snapshot(f"🧹 After cleanup tile {p}")
        
        if p >= 2:  # Only show first 3 tiles in detail
            print(f"    {Colors.OKCYAN}⏭️  ... (remaining {num_tiles - p - 1} tiles processed similarly){Colors.ENDC}")
            break
    
    return ga

def analyze_triton_sum_over_time(G: torch.Tensor, profiler: DetailedMemoryProfiler):
    """Analyze the sum over time norm squared computation"""
    B, T, d_g = G.shape
    
    print(f"    {Colors.OKCYAN}📊 Computing sum over time norm squared...{Colors.ENDC}")
    print(f"    📏 Input shape: {G.shape}")
    
    # This is a simpler operation - just sum of squares over time dimension
    gg = torch.sum(G * G, dim=1)  # [B, d_g]
    profiler.snapshot(f"🔢 After element-wise square and sum [{B}, {d_g}]")
    
    gg = torch.sum(gg, dim=1)  # [B]
    profiler.snapshot(f"📊 After final sum to [{B}]")
    
    return gg

def run_detailed_analysis():
    """Run detailed memory analysis with specified parameters"""
    print_header("DETAILED MEMORY ANALYSIS: Line-by-Line Memory Profiling", width=120, style="double")
    
    # Test parameters as specified
    B, T, d, p = 8, 16384, 2048, 2048  # Note: Using 16384 instead of 16834 for cleaner tile division
    tile_size = 64
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_icon = "🖥️" if device == 'cpu' else "🚀"
    print(f"{Colors.BOLD}🔧 Using device: {device_icon} {device}{Colors.ENDC}")
    print(f"{Colors.BOLD}📊 Test parameters: B={B}, T={T}, d={d}, p={p}{Colors.ENDC}")
    print(f"{Colors.BOLD}🔧 Tile size: {tile_size}{Colors.ENDC}")
    
    # Calculate theoretical memory requirements
    activation_memory = B * T * d * 4 / 1024 / 1024
    gradient_memory = B * T * p * 4 / 1024 / 1024
    gram_memory = B * T * T * 4 / 1024 / 1024 * 2
    tile_memory = B * tile_size * tile_size * 4 / 1024 / 1024 * 2
    
    print_section("THEORETICAL MEMORY REQUIREMENTS")
    print(f"  📈 Activations: {format_memory(activation_memory)}")
    print(f"  📉 Gradients: {format_memory(gradient_memory)}")
    print(f"  {Colors.WARNING}💥 Ghost Clipping method Gram matrices: {format_memory(gram_memory)}{Colors.ENDC}")
    print(f"  {Colors.OKGREEN}🔧 Triton method tile matrices: {format_memory(tile_memory)}{Colors.ENDC}")
    print(f"  {Colors.BOLD}{Colors.OKGREEN}💾 Memory savings ratio: {gram_memory / tile_memory:.2f}x{Colors.ENDC}")
    
    try:
        # Create test data
        activations, gradients, layer = create_test_data(B, T, d, p, device=device)
        
        # Analyze Ghost Clipping method
        ghost_clipping_result = analyze_ghost_clipping_method(layer, activations, gradients)
        
        if TRITON_AVAILABLE:
            # Clean up before Triton analysis
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Analyze Triton method
            triton_result = analyze_triton_method(layer, activations, gradients, tile_size=tile_size)
            
            # Verify results match
            print_header("ACCURACY VERIFICATION", style="single")
            
            for param_name in ['weight', 'bias']:
                param = getattr(layer, param_name)
                if param is not None and param.requires_grad:
                    ghost_clipping_norm = ghost_clipping_result[param]
                    triton_norm = triton_result[param]
                    
                    max_diff = torch.max(torch.abs(ghost_clipping_norm - triton_norm)).item()
                    rel_error = (max_diff / torch.max(ghost_clipping_norm).item()) if torch.max(ghost_clipping_norm).item() > 0 else 0
                    
                    status_icon = "✅" if max_diff < 1e-5 else "❌"
                    status_color = Colors.OKGREEN if max_diff < 1e-5 else Colors.FAIL
                    
                    print(f"{Colors.BOLD}🔍 {param_name.capitalize()}:{Colors.ENDC}")
                    print(f"  📊 Max absolute difference: {Colors.OKCYAN}{max_diff:.2e}{Colors.ENDC}")
                    print(f"  📈 Relative error: {Colors.OKCYAN}{rel_error:.2e}{Colors.ENDC}")
                    print(f"  {status_color}{status_icon} Status: {'PASSED' if max_diff < 1e-5 else 'FAILED'}{Colors.ENDC}")
        else:
            print_header("TRITON ANALYSIS SKIPPED", style="single")
            print(f"{Colors.WARNING}⚠️  Triton not available - Only Ghost Clipping method analysis was performed.{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error during analysis: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_detailed_analysis()