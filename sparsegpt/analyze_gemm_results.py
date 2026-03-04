"""
GEMM Hook 评估结果分析脚本
用于分析和可视化 GEMM Hook 评估的结果
"""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


def analyze_results(output_dir):
    """
    分析 GEMM Hook 评估结果
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"输出目录不存在: {output_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"GEMM Hook 评估结果分析")
    print(f"{'='*80}\n")
    
    # 统计信息
    results_summary = defaultdict(lambda: {
        'ppl_list': [],
        'gemm_counts': [],
    })
    
    # 遍历所有评估目录
    for eval_dir in sorted(output_dir.iterdir()):
        if not eval_dir.is_dir():
            continue
        
        # 检查是否有 evaluation_summary.json
        summary_file = eval_dir / 'evaluation_summary.json'
        if not summary_file.exists():
            continue
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        model = summary.get('model', 'unknown')
        config = summary.get('config', 'unknown')
        dataset = summary.get('dataset', 'unknown')
        
        key = f"{model:20s} | {config:10s} | {dataset:10s}"
        
        ppl_results = summary.get('ppl_results', [])
        avg_ppl = summary.get('average_ppl', 0)
        total_inferences = summary.get('total_inferences', 0)
        
        results_summary[key]['ppl_list'] = ppl_results
        results_summary[key]['avg_ppl'] = avg_ppl
        results_summary[key]['total_inferences'] = total_inferences
        
        # 收集 GEMM 操作计数
        gemm_count = 0
        for inf_idx in range(1, total_inferences + 1):
            inf_dir = eval_dir / f'inference_{inf_idx:03d}'
            gemm_file = inf_dir / 'gemm_stats.json'
            if gemm_file.exists():
                with open(gemm_file, 'r') as f:
                    gemm_stats = json.load(f)
                gemm_count = len(gemm_stats)
                results_summary[key]['gemm_counts'].append(gemm_count)
                break
    
    # 打印结果汇总
    print(f"{'模型':<20} | {'配置':<10} | {'数据集':<10} | {'平均PPL':>10} | {'推理次数':>6} | {'GEMM操作数':>8}")
    print("-" * 80)
    
    for key in sorted(results_summary.keys()):
        info = results_summary[key]
        avg_ppl = info.get('avg_ppl', 0)
        total_inf = info.get('total_inferences', 0)
        gemm_count = info['gemm_counts'][0] if info['gemm_counts'] else 0
        
        print(f"{key} | {avg_ppl:10.4f} | {total_inf:6d} | {gemm_count:8d}")
    
    print("\n" + "="*80)
    print("详细统计信息\n")
    
    # 按配置分组统计
    config_stats = defaultdict(lambda: {
        'ppl_all': [],
        'count': 0,
    })
    
    for key, info in results_summary.items():
        # 从 key 中提取配置信息
        parts = key.split('|')
        if len(parts) >= 2:
            config = parts[1].strip()
            config_stats[config]['ppl_all'].extend(info.get('ppl_list', []))
            config_stats[config]['count'] += 1
    
    # 打印配置级别统计
    print(f"{'配置':<15} | {'数据集数':>6} | {'总PPL数':>8} | {'平均PPL':>10} | {'最小PPL':>10} | {'最大PPL':>10}")
    print("-" * 75)
    
    for config in sorted(config_stats.keys()):
        ppl_list = config_stats[config]['ppl_all']
        count = config_stats[config]['count']
        
        if ppl_list:
            avg_ppl = statistics.mean(ppl_list)
            min_ppl = min(ppl_list)
            max_ppl = max(ppl_list)
            print(f"{config:<15} | {count:6d} | {len(ppl_list):8d} | {avg_ppl:10.4f} | {min_ppl:10.4f} | {max_ppl:10.4f}")
    
    # 目录结构说明
    print("\n" + "="*80)
    print("输出目录结构:\n")
    print(f"  {output_dir}/")
    print(f"    ├── opt-125m_config_dataset1/")
    print(f"    │   ├── evaluation_summary.json         # 该配置的评估摘要")
    print(f"    │   ├── inference_001/")
    print(f"    │   │   ├── gemm_stats.json             # GEMM 操作统计")
    print(f"    │   │   └── metadata.json               # 元数据（PPL 等）")
    print(f"    │   ├── inference_002/")
    print(f"    │   └── ... inference_010/")
    print(f"    ├── opt-125m_config_dataset2/")
    print(f"    └── ... (remaining configs × datasets)")
    
    print("\n" + "="*80)
    print(f"分析完成！共发现 {len(results_summary)} 个评估组合\n")


def export_csv(output_dir, csv_file):
    """
    导出结果为 CSV 格式
    """
    output_dir = Path(output_dir)
    csv_file = Path(csv_file)
    
    results = []
    
    for eval_dir in sorted(output_dir.iterdir()):
        if not eval_dir.is_dir():
            continue
        
        summary_file = eval_dir / 'evaluation_summary.json'
        if not summary_file.exists():
            continue
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        model = summary.get('model', 'unknown')
        config = summary.get('config', 'unknown')
        dataset = summary.get('dataset', 'unknown')
        
        for idx, ppl in enumerate(summary.get('ppl_results', []), 1):
            results.append({
                'model': model,
                'config': config,
                'dataset': dataset,
                'inference_id': idx,
                'ppl': ppl,
            })
    
    if results:
        import csv
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'config', 'dataset', 'inference_id', 'ppl'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"已导出结果到: {csv_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析 GEMM Hook 评估结果')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='评估输出目录')
    parser.add_argument('--export-csv', type=str, default=None,
                        help='导出结果到 CSV 文件')
    
    args = parser.parse_args()
    
    analyze_results(args.output_dir)
    
    if args.export_csv:
        export_csv(args.output_dir, args.export_csv)
