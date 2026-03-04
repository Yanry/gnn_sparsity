"""
测试 OPT Attention Hook - 验证能否捕获 QK^T 和 attention@V 的 matmul
"""
import torch
from transformers import OPTForCausalLM
from opt_attention_hook import OPTAttentionHook
import os


def test_opt_attention_hook():
    """测试 OPT Attention Hook"""
    
    print("="*80)
    print("测试 OPT Attention Hook")
    print("="*80)
    print()
    
    # 1. 创建 hook
    print("步骤 1: 创建 OPTAttentionHook")
    hook = OPTAttentionHook(save_tensors=False)
    
    # 2. 安装 hook
    print("步骤 2: 安装 hook")
    hook.install_hook()
    print()
    
    # 3. 加载模型
    print("步骤 3: 加载 OPT-125m 模型")
    # 强制使用 eager attention 实现（而不是 sdpa）
    model = OPTForCausalLM.from_pretrained('facebook/opt-125m', attn_implementation="eager")
    model.eval()
    print(f"  模型配置: {model.config.hidden_size} hidden, {model.config.num_attention_heads} heads, attn={model.config._attn_implementation}")
    print()
    
    # 4. 准备输入
    print("步骤 4: 准备输入数据")
    batch_size = 1
    seq_len = 128
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    print(f"  输入 shape: {input_ids.shape}")
    print()
    
    # 5. 运行推理
    print("步骤 5: 运行推理")
    with torch.no_grad():
        outputs = model(input_ids)
    print(f"  输出 shape: {outputs.logits.shape}")
    print()
    
    # 6. 查看捕获的数据
    print("步骤 6: 查看捕获的数据")
    hook.print_summary()
    
    # 7. 保存数据
    output_file = 'test_opt_attention_data.json'
    print(f"步骤 7: 保存数据到 {output_file}")
    hook.save_to_json(output_file)
    print()
    
    # 8. 验证关键数据
    print("步骤 8: 验证关键数据")
    
    # 检查是否捕获了所有层
    num_layers = model.config.num_hidden_layers
    print(f"  模型层数: {num_layers}")
    
    qkt_count = sum(1 for k in hook.data.keys() if 'QK^T' in k)
    attn_weights_count = sum(1 for k in hook.data.keys() if 'attention_weights' in k)
    attn_v_count = sum(1 for k in hook.data.keys() if 'attention@V' in k)
    
    print(f"  捕获的 Q@K^T 操作: {qkt_count}")
    print(f"  捕获的 softmax 操作: {attn_weights_count}")
    print(f"  捕获的 attention@V 操作: {attn_v_count}")
    
    # 验证形状
    print()
    print("  验证形状:")
    for layer_id in range(min(3, num_layers)):  # 只检查前3层
        qkt_key = f'layer_{layer_id}_QK^T'
        if qkt_key in hook.data:
            qkt_data = hook.data[qkt_key]
            expected_shape = [batch_size, model.config.num_attention_heads, seq_len, seq_len]
            actual_shape = qkt_data['output_shape']
            match = expected_shape == actual_shape
            print(f"    Layer {layer_id} QK^T: {actual_shape} {'✓' if match else '✗ 期望 ' + str(expected_shape)}")
    
    print()
    print("="*80)
    
    # 9. 移除 hook
    print("步骤 9: 移除 hook")
    hook.remove_hook()
    
    print()
    print("✓ 测试完成！")
    print()
    print("输出文件:")
    print(f"  - {output_file}")
    
    return hook


if __name__ == "__main__":
    test_opt_attention_hook()
