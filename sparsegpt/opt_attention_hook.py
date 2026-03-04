"""
OPT Attention Hook - 直接 Hook eager_attention_forward 来捕获 QK^T 和 attention@V 的 matmul
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import json
import os


class OPTAttentionHook:
    """
    用于 Hook OPT 模型的 attention 计算，捕获：
    1. Q @ K^T 的 matmul 操作
    2. softmax(Q@K^T) 的结果
    3. softmax(Q@K^T) @ V 的 matmul 操作
    """
    
    def __init__(self, save_tensors=False):
        """
        Args:
            save_tensors: 是否保存 tensor 数据（会占用大量内存）
        """
        self.save_tensors = save_tensors
        self.data = {}
        self.layer_counter = 0
        self.original_forward = None

    def _safe_quantile(self, tensor: torch.Tensor, q: float, max_samples: int = 2_000_000) -> float:
        """在大张量上安全计算分位数：超出阈值时做等步长采样，避免 quantile OOM/size 错误。"""
        stats_tensor = tensor.detach().float().reshape(-1)
        numel = stats_tensor.numel()
        if numel == 0:
            return 0.0

        if numel > max_samples:
            step = max(1, numel // max_samples)
            stats_tensor = stats_tensor[::step]
            if stats_tensor.numel() > max_samples:
                stats_tensor = stats_tensor[:max_samples]

        # 在 CPU 上计算更稳妥
        stats_tensor = stats_tensor.cpu()
        return float(torch.quantile(stats_tensor, q).item())
        
    def create_hooked_attention_forward(self):
        """
        创建一个 hooked 版本的 eager_attention_forward
        """
        # 导入原始函数
        from transformers.models.opt.modeling_opt import eager_attention_forward
        self.original_forward = eager_attention_forward
        
        # 在闭包外部定义变量，供闭包内访问
        hook_data = self.data
        save_tensors_flag = self.save_tensors
        call_counter = [0]  # 使用列表以便在闭包中修改
        
        # 创建 wrapper
        def hooked_eager_attention_forward(
            module: nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float,
            dropout: float = 0.0,
            **kwargs,
        ):
            """
            Hooked 版本的 eager_attention_forward
            在原始实现基础上记录所有中间结果
            """
            layer_id = getattr(module, 'layer_idx', call_counter[0])
            call_counter[0] += 1
            
            # ===== 捕获 Q @ K^T matmul =====
            key_transposed = key.transpose(-1, -2)
            attn_weights_raw = torch.matmul(query, key_transposed)
            attn_weights = attn_weights_raw * scaling
            
            # 记录 QK^T 的信息
            hook_data[f'layer_{layer_id}_QK^T'] = {
                'operation': 'Q @ K^T',
                'query_shape': list(query.shape),
                'key_shape': list(key.shape),
                'key_transposed_shape': list(key_transposed.shape),
                'output_shape': list(attn_weights.shape),
                'scaling_factor': float(scaling),
                'output_numel': int(attn_weights.numel()),
                'output_sparsity': float((attn_weights == 0).sum().item() / attn_weights.numel()),
                'output_min': float(attn_weights.min().item()),
                'output_max': float(attn_weights.max().item()),
                'output_mean': float(attn_weights.mean().item()),
                'output_std': float(attn_weights.std().item()),
            }
            
            if save_tensors_flag:
                hook_data[f'layer_{layer_id}_QK^T']['tensor'] = attn_weights.detach().cpu()
            
            # 应用 attention mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # ===== 捕获 softmax 操作 =====
            attn_weights_softmax = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            try:
                top_1_percent_threshold = self._safe_quantile(attn_weights_softmax, 0.99)
                top_10_percent_threshold = self._safe_quantile(attn_weights_softmax, 0.90)
            except Exception:
                top_1_percent_threshold = None
                top_10_percent_threshold = None
            
            # 记录 softmax 后的 attention weights
            hook_data[f'layer_{layer_id}_attention_weights'] = {
                'operation': 'softmax(Q@K^T)',
                'input_shape': list(attn_weights.shape),
                'output_shape': list(attn_weights_softmax.shape),
                'output_numel': int(attn_weights_softmax.numel()),
                'output_sparsity': float((attn_weights_softmax == 0).sum().item() / attn_weights_softmax.numel()),
                'output_min': float(attn_weights_softmax.min().item()),
                'output_max': float(attn_weights_softmax.max().item()),
                'output_mean': float(attn_weights_softmax.mean().item()),
                'output_std': float(attn_weights_softmax.std().item()),
                'top_1_percent_threshold': top_1_percent_threshold,
                'top_10_percent_threshold': top_10_percent_threshold,
            }
            
            if save_tensors_flag:
                hook_data[f'layer_{layer_id}_attention_weights']['tensor'] = attn_weights_softmax.detach().cpu()
            
            # 应用 dropout
            attn_weights_dropped = F.dropout(attn_weights_softmax, p=dropout, training=module.training)
            
            # ===== 捕获 softmax(QK^T) @ V matmul =====
            attn_output = torch.matmul(attn_weights_dropped, value)
            
            # 记录 attention @ V的信息
            hook_data[f'layer_{layer_id}_attention@V'] = {
                'operation': 'softmax(Q@K^T) @ V',
                'attention_weights_shape': list(attn_weights_dropped.shape),
                'value_shape': list(value.shape),
                'output_shape': list(attn_output.shape),
                'output_numel': int(attn_output.numel()),
                'output_sparsity': float((attn_output == 0).sum().item() / attn_output.numel()),
                'output_min': float(attn_output.min().item()),
                'output_max': float(attn_output.max().item()),
                'output_mean': float(attn_output.mean().item()),
                'output_std': float(attn_output.std().item()),
            }
            
            if save_tensors_flag:
                hook_data[f'layer_{layer_id}_attention@V']['tensor'] = attn_output.detach().cpu()
            
            # transpose 回原始形状
            attn_output = attn_output.transpose(1, 2).contiguous()
            
            return attn_output, attn_weights_softmax
        
        return hooked_eager_attention_forward
    
    def install_hook(self):
        """
        安装 hook - 替换 transformers 中的 eager_attention_forward
        """
        import transformers.models.opt.modeling_opt as opt_modeling
        
        # 保存原始函数
        self.original_forward = opt_modeling.eager_attention_forward
        
        # 替换为 hooked 版本
        opt_modeling.eager_attention_forward = self.create_hooked_attention_forward()
        
        print("✓ 已安装 OPT Attention Hook")
    
    def remove_hook(self):
        """
        移除 hook - 恢复原始的 eager_attention_forward
        """
        if self.original_forward is not None:
            import transformers.models.opt.modeling_opt as opt_modeling
            opt_modeling.eager_attention_forward = self.original_forward
            print("✓ 已移除 OPT Attention Hook")
    
    def reset(self):
        """重置捕获的数据"""
        self.data = {}
        self.layer_counter = 0
    
    def get_summary(self):
        """获取捕获数据的摘要"""
        summary = {}
        
        for key, value in sorted(self.data.items()):
            # 移除 tensor 数据（太大）
            summary[key] = {k: v for k, v in value.items() if k != 'tensor'}
        
        return summary
    
    def save_to_json(self, filepath):
        """保存数据到 JSON 文件"""
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ 已保存 attention 数据到: {filepath}")
        print(f"  - 捕获的操作数: {len(summary)}")
    
    def print_summary(self):
        """打印摘要信息"""
        print("\n" + "="*80)
        print("OPT Attention Hook 摘要")
        print("="*80)
        
        # 统计每层的信息
        layers = set()
        for key in self.data.keys():
            if 'layer_' in key:
                layer_id = key.split('_')[1]
                layers.add(int(layer_id))
        
        print(f"捕获的层数: {len(layers)}")
        print(f"总操作数: {len(self.data)}")
        print()
        
        # 打印每层的统计
        for layer_id in sorted(layers):
            print(f"Layer {layer_id}:")
            
            # QK^T
            qkt_key = f'layer_{layer_id}_QK^T'
            if qkt_key in self.data:
                qkt = self.data[qkt_key]
                print(f"  Q@K^T: shape={qkt['output_shape']}, "
                      f"range=[{qkt['output_min']:.4f}, {qkt['output_max']:.4f}], "
                      f"mean={qkt['output_mean']:.4f}")
            
            # Attention weights
            attn_key = f'layer_{layer_id}_attention_weights'
            if attn_key in self.data:
                attn = self.data[attn_key]
                print(f"  Softmax: shape={attn['output_shape']}, "
                      f"range=[{attn['output_min']:.6f}, {attn['output_max']:.6f}], "
                      f"mean={attn['output_mean']:.6f}")
            
            # Attention @ V
            attn_v_key = f'layer_{layer_id}_attention@V'
            if attn_v_key in self.data:
                attn_v = self.data[attn_v_key]
                print(f"  Attn@V: shape={attn_v['output_shape']}, "
                      f"range=[{attn_v['output_min']:.4f}, {attn_v['output_max']:.4f}], "
                      f"mean={attn_v['output_mean']:.4f}")
            
            print()


if __name__ == "__main__":
    print("OPTAttentionHook 已准备就绪")
    print("使用示例见 test_opt_attention_hook.py")
