"""
直接在 OPT 模型的 attention 前向传播中捕获 Q/K/V 和 QK^T matmul 的信息
这样可以绕过 torch.matmul hook 的问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class AttentionDataCapture:
    """
    用于捕获 Attention 中间数据的工具类
    包括 Q/K/V 投影输出、QK^T、softmax(QK^T) 等
    """
    
    def __init__(self):
        self.data = defaultdict(dict)
        self.hooks = []
    
    def register_attention_capture_hooks(self, model):
        """
        为模型的所有 attention 层注册 hook，捕获 Q/K/V 投影和 matmul 信息
        """
        layer_count = 0
        
        for name, module in model.named_modules():
            if 'self_attn' in name and not 'layer_norm' in name:
                # 为 attention 的 forward 注册 hook
                block_id = layer_count
                
                # Hook the forward method to capture intermediate results
                def create_attn_forward_hook(bid):
                    """创建 attention forward hook"""
                    original_forward = None
                    
                    def forward_hook(module, input, output):
                        """捕获 attention 的中间计算过程"""
                        try:
                            hidden_states = input[0]
                            
                            # 获取 Q/K/V 投影输出
                            Q = module.q_proj(hidden_states)
                            K = module.k_proj(hidden_states)
                            V = module.v_proj(hidden_states)
                            
                            batch_size, seq_len, hidden_dim = Q.shape
                            num_heads = 12  # OPT-125m 固定为 12
                            head_dim = hidden_dim // num_heads
                            
                            # 保存投影层输出信息
                            self.data[f'layer_{bid}']['Q_proj'] = {
                                'shape': tuple(Q.shape),
                                'sparsity': float((Q == 0).sum() / Q.numel()),
                                'numel': int(Q.numel()),
                            }
                            self.data[f'layer_{bid}']['K_proj'] = {
                                'shape': tuple(K.shape),
                                'sparsity': float((K == 0).sum() / K.numel()),
                                'numel': int(K.numel()),
                            }
                            self.data[f'layer_{bid}']['V_proj'] = {
                                'shape': tuple(V.shape),
                                'sparsity': float((V == 0).sum() / V.numel()),
                                'numel': int(V.numel()),
                            }
                            
                            # ===== 关键：计算 QK^T matmul =====
                            # 将 Q/K/V reshape 为多头形式
                            # OPT 中通常的处理方式
                            
                            # 方式1: 如果模型内部处理多头（推荐）
                            # reshape Q: [batch, seq_len, hidden_dim] -> [batch, seq_len, num_heads, head_dim]
                            Q_reshaped = Q.view(batch_size, seq_len, num_heads, head_dim)
                            K_reshaped = K.view(batch_size, seq_len, num_heads, head_dim)
                            V_reshaped = V.view(batch_size, seq_len, num_heads, head_dim)
                            
                            # transpose: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
                            Q_reshaped = Q_reshaped.transpose(1, 2)
                            K_reshaped = K_reshaped.transpose(1, 2)
                            V_reshaped = V_reshaped.transpose(1, 2)
                            
                            # 计算 QK^T (对于每个 head 分别计算)
                            # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
                            # = [batch, num_heads, seq_len, seq_len]
                            
                            qkt = torch.matmul(Q_reshaped, K_reshaped.transpose(-2, -1))
                            qkt = qkt / (head_dim ** 0.5)  # scaled
                            
                            self.data[f'layer_{bid}']['QK^T'] = {
                                'shape': tuple(qkt.shape),
                                'expected_shape': (batch_size, num_heads, seq_len, seq_len),
                                'sparsity': float((qkt == 0).sum() / qkt.numel()),
                                'numel': int(qkt.numel()),
                                'min_value': float(qkt.min().item()),
                                'max_value': float(qkt.max().item()),
                                'mean_value': float(qkt.mean().item()),
                            }
                            
                            # 计算 softmax(QK^T)
                            attn_weights = F.softmax(qkt, dim=-1)
                            
                            self.data[f'layer_{bid}']['attention_weights'] = {
                                'shape': tuple(attn_weights.shape),
                                'sparsity': float((attn_weights == 0).sum() / attn_weights.numel()),
                                'numel': int(attn_weights.numel()),
                                'min_value': float(attn_weights.min().item()),
                                'max_value': float(attn_weights.max().item()),
                                'mean_value': float(attn_weights.mean().item()),
                            }
                            
                            # ===== 关键：计算 softmax(QK^T) @ V =====
                            attn_output_raw = torch.matmul(attn_weights, V_reshaped)
                            # attn_output_raw: [batch, num_heads, seq_len, head_dim]
                            
                            self.data[f'layer_{bid}']['attention_output'] = {
                                'shape': tuple(attn_output_raw.shape),
                                'expected_shape': (batch_size, num_heads, seq_len, head_dim),
                                'sparsity': float((attn_output_raw == 0).sum() / attn_output_raw.numel()),
                                'numel': int(attn_output_raw.numel()),
                                'min_value': float(attn_output_raw.min().item()),
                                'max_value': float(attn_output_raw.max().item()),
                                'mean_value': float(attn_output_raw.mean().item()),
                            }
                            
                            # 重拼接为原始形状
                            attn_output_final = attn_output_raw.transpose(1, 2).contiguous()
                            attn_output_final = attn_output_final.view(batch_size, seq_len, hidden_dim)
                            
                            # out_proj
                            out = module.out_proj(attn_output_final)
                            
                            self.data[f'layer_{bid}']['final_output'] = {
                                'shape': tuple(out.shape),
                                'sparsity': float((out == 0).sum() / out.numel()),
                                'numel': int(out.numel()),
                            }
                            
                        except Exception as e:
                            print(f"Error in attention capture for layer {bid}: {e}")
                    
                    return forward_hook
                
                # 注册 hook
                hook = module.register_forward_hook(create_attn_forward_hook(block_id))
                self.hooks.append(hook)
                layer_count += 1
    
    def get_summary(self):
        """获取捕获数据的总结"""
        summary = {}
        
        for layer_name, layer_data in sorted(self.data.items()):
            summary[layer_name] = {}
            for op_name, op_data in layer_data.items():
                summary[layer_name][op_name] = {
                    'shape': op_data.get('shape'),
                    'sparsity': op_data.get('sparsity', 0),
                }
                if 'expected_shape' in op_data:
                    summary[layer_name][op_name]['expected_shape'] = op_data['expected_shape']
        
        return summary
    
    def remove_hooks(self):
        """移除所有注册的 hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def save_to_json(self, filepath):
        """保存捕获的数据到 JSON 文件"""
        import json
        
        # 将数据转换为可序列化的格式
        serializable_data = {}
        for layer_name, layer_data in self.data.items():
            serializable_data[layer_name] = {}
            for op_name, op_data in layer_data.items():
                serializable_data[layer_name][op_name] = {}
                for key, value in op_data.items():
                    if isinstance(value, (list, tuple)):
                        serializable_data[layer_name][op_name][key] = list(value)
                    else:
                        serializable_data[layer_name][op_name][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"已保存 attention 数据到: {filepath}")


# 使用示例
if __name__ == "__main__":
    """
    使用示例：
    
    from transformers import OPTForCausalLM
    
    model = OPTForCausalLM.from_pretrained('facebook/opt-125m')
    
    # 创建捕获工具
    capture = AttentionDataCapture()
    capture.register_attention_capture_hooks(model)
    
    # 运行模型
    with torch.no_grad():
        outputs = model(input_ids)
    
    # 获取数据
    summary = capture.get_summary()
    print(summary)
    
    # 保存数据
    capture.save_to_json('attention_data.json')
    """
    
    print("AttentionDataCapture 模块已准备就绪")
    print("详见该文件的使用示例和 AttentionDataCapture 类")
