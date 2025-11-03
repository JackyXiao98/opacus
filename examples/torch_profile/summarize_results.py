#!/usr/bin/env python3
"""
结果汇总脚本 - 分析profiling实验的日志文件并生成汇总报告
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def parse_log_file(log_path: str) -> Dict[str, Any]:
    """解析单个日志文件，提取关键信息"""
    result = {
        "config": {},
        "status": "unknown",
        "model_params": 0,
        "memory_usage": {},
        "errors": [],
        "completion_time": None
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取配置信息
        config_match = re.search(r'Trainer: (\w+), Batch Size: (\d+), Seq Length: (\d+), Model: (\w+)', content)
        if config_match:
            result["config"] = {
                "trainer": config_match.group(1),
                "batch_size": int(config_match.group(2)),
                "seq_length": int(config_match.group(3)),
                "model_size": config_match.group(4)
            }
        
        # 提取模型参数数量
        params_match = re.search(r'Model parameters: ([\d,]+) \(([\d.]+)M\)', content)
        if params_match:
            result["model_params"] = int(params_match.group(1).replace(',', ''))
        
        # 提取内存使用信息
        memory_patterns = {
            "initial": r'Initial Memory Usage:.*?GPU Allocated: ([\d.]+) MB.*?System RSS: ([\d.]+) MB',
            "after_model": r'After model creation Memory Usage:.*?GPU Allocated: ([\d.]+) MB.*?System RSS: ([\d.]+) MB',
            "after_profiling": r'After profiling Memory Usage:.*?GPU Allocated: ([\d.]+) MB.*?System RSS: ([\d.]+) MB',
            "after_cleanup": r'After cleanup Memory Usage:.*?GPU Allocated: ([\d.]+) MB.*?System RSS: ([\d.]+) MB'
        }
        
        for stage, pattern in memory_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result["memory_usage"][stage] = {
                    "gpu_allocated_mb": float(match.group(1)),
                    "system_rss_mb": float(match.group(2))
                }
        
        # 检查完成状态
        if "Single configuration profiling completed successfully" in content:
            result["status"] = "success"
        elif "ERROR" in content or "Exception" in content or "Failed" in content:
            result["status"] = "failed"
            # 提取错误信息
            error_matches = re.findall(r'ERROR.*?(?=\n|$)', content)
            result["errors"] = error_matches[:5]  # 最多保留5个错误
        
        # 提取完成时间（如果有的话）
        time_match = re.search(r'Step \d+, Loss: ([\d.]+)', content)
        if time_match:
            result["completion_time"] = datetime.now().isoformat()
    
    except Exception as e:
        result["status"] = "parse_error"
        result["errors"] = [f"Failed to parse log file: {str(e)}"]
    
    return result


def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成汇总报告"""
    summary = {
        "total_experiments": len(results),
        "successful": 0,
        "failed": 0,
        "parse_errors": 0,
        "by_trainer": {},
        "memory_analysis": {},
        "failed_experiments": []
    }
    
    for result in results:
        # 统计状态
        if result["status"] == "success":
            summary["successful"] += 1
        elif result["status"] == "failed":
            summary["failed"] += 1
            summary["failed_experiments"].append({
                "config": result["config"],
                "errors": result["errors"]
            })
        else:
            summary["parse_errors"] += 1
        
        # 按trainer分类统计
        trainer = result["config"].get("trainer", "unknown")
        if trainer not in summary["by_trainer"]:
            summary["by_trainer"][trainer] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "avg_model_params": 0,
                "memory_usage": []
            }
        
        summary["by_trainer"][trainer]["total"] += 1
        if result["status"] == "success":
            summary["by_trainer"][trainer]["successful"] += 1
        elif result["status"] == "failed":
            summary["by_trainer"][trainer]["failed"] += 1
        
        # 收集内存使用数据
        if result["memory_usage"]:
            summary["by_trainer"][trainer]["memory_usage"].append(result["memory_usage"])
    
    # 计算平均内存使用
    for trainer_data in summary["by_trainer"].values():
        if trainer_data["memory_usage"]:
            # 计算平均GPU内存使用
            gpu_usage = []
            for mem_data in trainer_data["memory_usage"]:
                if "after_profiling" in mem_data:
                    gpu_usage.append(mem_data["after_profiling"]["gpu_allocated_mb"])
            
            if gpu_usage:
                trainer_data["avg_gpu_memory_mb"] = sum(gpu_usage) / len(gpu_usage)
    
    return summary


def print_summary_report(summary: Dict[str, Any]):
    """打印汇总报告"""
    print("=" * 60)
    print("🎯 PROFILING EXPERIMENTS SUMMARY REPORT")
    print("=" * 60)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 总体统计
    print("📊 Overall Statistics:")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Successful: {summary['successful']} ({summary['successful']/summary['total_experiments']*100:.1f}%)")
    print(f"  Failed: {summary['failed']} ({summary['failed']/summary['total_experiments']*100:.1f}%)")
    if summary['parse_errors'] > 0:
        print(f"  Parse errors: {summary['parse_errors']}")
    print()
    
    # 按trainer统计
    print("🔍 Results by Trainer:")
    for trainer, data in summary["by_trainer"].items():
        success_rate = data["successful"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"  {trainer}:")
        print(f"    Success rate: {data['successful']}/{data['total']} ({success_rate:.1f}%)")
        if "avg_gpu_memory_mb" in data:
            print(f"    Avg GPU memory: {data['avg_gpu_memory_mb']:.1f} MB")
    print()
    
    # 失败的实验
    if summary["failed_experiments"]:
        print("❌ Failed Experiments:")
        for i, failed in enumerate(summary["failed_experiments"][:5], 1):  # 只显示前5个
            config = failed["config"]
            print(f"  {i}. {config.get('trainer', 'unknown')} "
                  f"(bs={config.get('batch_size', '?')}, seq={config.get('seq_length', '?')})")
            if failed["errors"]:
                print(f"     Error: {failed['errors'][0][:100]}...")
        
        if len(summary["failed_experiments"]) > 5:
            print(f"     ... and {len(summary['failed_experiments']) - 5} more")
        print()
    
    print("📋 For detailed results, check individual log files in ./logs/")
    print("📊 View TensorBoard results: tensorboard --logdir=./runs")


def main():
    parser = argparse.ArgumentParser(description="Summarize profiling experiment results")
    parser.add_argument("--logs-dir", type=str, default="logs",
                       help="Directory containing log files")
    parser.add_argument("--output", type=str, 
                       help="Output JSON file for detailed results")
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Error: Logs directory '{logs_dir}' not found")
        return 1
    
    # 查找所有日志文件
    log_files = list(logs_dir.glob("*.log"))
    if not log_files:
        print(f"No log files found in '{logs_dir}'")
        return 1
    
    print(f"Found {len(log_files)} log files to analyze...")
    
    # 解析所有日志文件
    results = []
    for log_file in log_files:
        print(f"Parsing {log_file.name}...")
        result = parse_log_file(str(log_file))
        result["log_file"] = str(log_file)
        results.append(result)
    
    # 生成汇总报告
    summary = generate_summary_report(results)
    
    # 打印报告
    print_summary_report(summary)
    
    # 保存详细结果到JSON文件
    if args.output:
        output_data = {
            "summary": summary,
            "detailed_results": results,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Detailed results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())