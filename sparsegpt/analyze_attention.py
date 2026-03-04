"""
分析 GEMM stats 中的 Attention 操作信息
展示 Q/K/V 的 shape、QK^T 的 shape、以及 softmax(QK^T) @ V 的信息
"""
import json
import os
from collections import defaultdict
import numpy as np


def analyze_attention_operations(gemm_stats_path):
    """
    分析 GEMM stats JSON 中的 Attention 操作
    """
    with open(gemm_stats_path, 'r') as f:
        gemm_data = json.load(f)
    
    print("=" * 80)
    print("ATTENTION 操作分析")
    print("=" * 80)
    
    # 1. 分析投影层输出 (Q/K/V/out_proj)
    print("\n[1] 投影层输出分析")
    print("-" * 80)
    
    projection_info = defaultdict(dict)
    for key, value in gemm_data.items():
        if 'projection' in value.get('operation', ''):
            parts = key.split('_')
            if 'q_proj' in key:
                block_id = key.split('_')[1]
                projection_info[block_id]['q_output_shape'] = value['output_shape']
            elif 'k_proj' in key:
                block_id = key.split('_')[1]
                projection_info[block_id]['k_output_shape'] = value['output_shape']
            elif 'v_proj' in key:
                block_id = key.split('_')[1]
                projection_info[block_id]['v_output_shape'] = value['output_shape']
    
    # 也分析Linear层中的 q_proj/k_proj/v_proj
    for key, value in gemm_data.items():
        if 'linear_' in key and 'q_proj' in key:
            print(f"Q投影({key}):")
            print(f"  输入 shape: {value['input_shape']}")
            print(f"  权重 shape: {value['weight_shape']}")
            print(f"  预期输出 shape: [{value['input_shape'][0]}, {value['input_shape'][1]}, {value['weight_shape'][0]}]")
            
            block_num = key.split('_')[0].replace('linear', '')
            seq_len = value['input_shape'][1]
            hidden_dim = value['weight_shape'][0]
            
            # 在OPT中，通常 hidden_dim = num_heads * head_dim
            # OPT-125m: hidden_dim=768, num_heads=12, head_dim=64
            num_heads = 12  # OPT-125m 固定
            head_dim = hidden_dim // num_heads
            
            print(f"  多头拆分: [{num_heads} heads, {head_dim} dim/head]")
            print()
        elif 'linear_' in key and 'k_proj' in key:
            print(f"K投影({key}):")
            print(f"  输入 shape: {value['input_shape']}")
            print(f"  权重 shape: {value['weight_shape']}")
            seq_len = value['input_shape'][1]
            print()
        elif 'linear_' in key and 'v_proj' in key:
            print(f"V投影({key}):")
            print(f"  输入 shape: {value['input_shape']}")
            print(f"  权重 shape: {value['weight_shape']}")
            print()
    
    # 2. 计算和展示 QK^T 的 shape
    print("\n[2] QK^T 操作的 Shape 计算")
    print("-" * 80)
    
    q_ops = {}
    k_ops = {}
    
    for key, value in gemm_data.items():
        if 'linear_' in key and 'q_proj' in key:
            output_shape = tuple([value['input_shape'][0], value['input_shape'][1], value['weight_shape'][0]])
            q_ops[key] = {
                'output_shape': output_shape,
                'seq_len': value['input_shape'][1],
                'hidden_dim': value['weight_shape'][0],
            }
        elif 'linear_' in key and 'k_proj' in key:
            output_shape = tuple([value['input_shape'][0], value['input_shape'][1], value['weight_shape'][0]])
            k_ops[key] = {
                'output_shape': output_shape,
                'seq_len': value['input_shape'][1],
                'hidden_dim': value['weight_shape'][0],
            }
    
    print(f"共找到 {len(q_ops)} 个 Q 投影和 {len(k_ops)} 个 K 投影")
    print()
    
    # 对应 Q 和 K 的投影
    for q_key in sorted(q_ops.keys()):
        k_key = q_key.replace('q_proj', 'k_proj')
        if k_key in k_ops:
            q_info = q_ops[q_key]
            k_info = k_ops[k_key]
            
            # Q @ K^T 的计算
            # Q: [batch, seq_len, hidden_dim] 
            # K: [batch, seq_len, hidden_dim]
            # 需要转置 K
            # QK^T: [batch, seq_len, seq_len]
            
            print(f"Layer {q_key.split('_')[1]} 中的 QK^T 操作:")
            print(f"  Q shape: {q_info['output_shape']}")
            print(f"  K shape: {k_info['output_shape']}")
            print(f"  K转置后: [{k_info['output_shape'][0]}, {k_info['output_shape'][2]}, {k_info['output_shape'][1]}]")
            
            batch = q_info['output_shape'][0]
            seq_len = q_info['output_shape'][1]
            qkt_shape = (batch, seq_len, seq_len)
            print(f"  QK^T shape: {qkt_shape}")
            print(f"  QK^T 元素数: {np.prod(qkt_shape):,}")
            
            # 多头情况下
            hidden_dim = q_info['hidden_dim']
            num_heads = 12
            head_dim = hidden_dim // num_heads
            print(f"  多头模式: {num_heads} heads × {head_dim} dimensions")
            print(f"  单个head的QK^T: [{batch}, {seq_len}, {seq_len}] (第num_heads维被处理)")
            print()
    
    # 3. 查找并分析 matmul 操作
    print("\n[3] Matmul 操作分析 (包含 QK^T 和 Attn@V)")
    print("-" * 80)
    
    matmul_ops = []
    for key, value in gemm_data.items():
        if 'matmul' in key or 'bmm' in key or 'mm' in key:
            matmul_ops.append((key, value))
    
    if matmul_ops:
        print(f"找到 {len(matmul_ops)} 个 matmul 操作:\n")
        
        for key, value in sorted(matmul_ops)[:20]:  # 显示前20个
            print(f"操作 {key}:")
            print(f"  Input1 shape: {value.get('input1_shape')}")
            print(f"  Input2 shape: {value.get('input2_shape')}")
            print(f"  Output shape: {value.get('output_shape')}")
            is_attn = value.get('is_attention_like', False)
            print(f"  是否为 Attention 相关: {is_attn}")
            print(f"  稀疏性: inp1={value.get('input1_sparsity', 0):.4f}, " +
                  f"inp2={value.get('input2_sparsity', 0):.4f}, " +
                  f"out={value.get('output_sparsity', 0):.4f}")
            print()
    else:
        print("⚠️ 没找到 matmul 操作记录")
        print("   这说明当前的 hook 可能没有正确捕获 matmul 操作")
        print("   可能的原因:")
        print("   1. OPT 使用了优化的 attention 实现 (如 scaled_dot_product_attention)")
        print("   2. Matmul hook 的替换没有生效")
        print("   3. 或者 attention 中的 matmul 被 inlined 了")
    
    # 4. 分析 attention 输出
    print("\n[4] Attention 输出分析")
    print("-" * 80)
    
    attn_outputs = {}
    for key, value in gemm_data.items():
        if 'attention_output' in value.get('operation', ''):
            attn_outputs[key] = value
    
    if attn_outputs:
        for key, value in sorted(attn_outputs.items()):
            print(f"{key}: shape={value['output_shape']}, sparsity={value['output_sparsity']:.6f}")
    else:
        print("没找到 attention_output 记录")
    
    # 5. 总结
    print("\n[5] 总结与建议")
    print("-" * 80)
    print("✓ 当前能捕获的:")
    print("  - 线性层 (q_proj, k_proj, v_proj, out_proj) 的输入输出")
    print("  - Attention 层的最终输出")
    print()
    print("✗ 当前不能捕获的:")
    print("  - Q @ K^T 的 matmul 操作及其输出")
    print("  - softmax(QK^T) @ V 的 matmul 操作及其输出")
    print("  - attention scores (softmax后的权重)")
    print()
    print("📌 建议:")
    print("  1. 使用改进的 gemm_hook_improved.py (支持更细致的投影输出捕获)")
    print("  2. 修改 OPT attention 的实现，直接 hook Q/K/V 的 matmul")
    print("  3. 可以通过 torch.jit.trace 或修改模型代码来捕获中间的 matmul")


def calculate_qkt_shape_from_config(hidden_size=768, num_heads=12, seq_len=2048, batch_size=1):
    """
    根据配置计算 QK^T 的形状
    
    对于 OPT-125m:
    - hidden_size: 768
    - num_attention_heads: 12
    - head_dim: 64
    - sequence_length: 2048 (默认)
    """
    head_dim = hidden_size // num_heads
    
    print("=" * 80)
    print("QK^T 形状计算（基于模型配置）")
    print("=" * 80)
    print(f"模型配置:")
    print(f"  隐层维度: {hidden_size}")
    print(f"  注意力头数: {num_heads}")
    print(f"  每头维度: {head_dim}")
    print(f"  序列长度: {seq_len}")
    print(f"  批处理大小: {batch_size}")
    print()
    
    print(f"Q 的形状:")
    print(f"  未拆分: [{batch_size}, {seq_len}, {hidden_size}]")
    print(f"  拆分为多头: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    print(f"  或重排为: [{batch_size}, {seq_len}, {num_heads}, {head_dim}]")
    print()
    
    print(f"K 的形状:")
    print(f"  未拆分: [{batch_size}, {seq_len}, {hidden_size}]")
    print(f"  拆分为多头: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    print()
    
    print(f"QK^T 的形状（每个 head）:")
    print(f"  [{batch_size}, {num_heads}, {seq_len}, {seq_len}]")
    print(f"  或（全部展开）: [{batch_size}, {seq_len}, {seq_len}]（实际计算在头维度上进行）")
    print()
    
    qkt_size = batch_size * seq_len * seq_len  # 单head
    print(f"单个 head 的 QK^T 元素数: {qkt_size:,}")
    print(f"全部 {num_heads} heads 的总元素数: {qkt_size * num_heads:,}")
    print()
    
    # Attention @ V
    print(f"softmax(QK^T) @ V 的形状:")
    print(f"  Attn scores: [{batch_size}, {num_heads}, {seq_len}, {seq_len}]")
    print(f"  V: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    print(f"  Output: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    print(f"  或重拼接后: [{batch_size}, {seq_len}, {hidden_size}]")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        gemm_stats_file = sys.argv[1]
        if os.path.exists(gemm_stats_file):
            analyze_attention_operations(gemm_stats_file)
        else:
            print(f"文件不存在: {gemm_stats_file}")
    else:
        # 默认分析
        print("用法: python analyze_attention.py <gemm_stats.json>")
        print()
        
        # 显示默认的 shape 计算
        calculate_qkt_shape_from_config()
