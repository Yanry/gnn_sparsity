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
        self._save_tensors_flag = save_tensors  # 重命名以避免与方法冲突
        self.data = {}
        self.tensor_cache = {}  # 存储所有输入tensor
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
        tensor_cache = self.tensor_cache  # 保存对tensor_cache的引用
        save_tensors_flag = self._save_tensors_flag
        safe_quantile = self._safe_quantile  # 保存method引用
        call_counter = [0]  # 使用列表以便在闭包中修改
        original_forward = self.original_forward  # 保存原始函数副本
        
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
            
            # ===== 捕获 Q @ K^T matmul 的输入 =====
            key_transposed = key.transpose(-1, -2)
            attn_weights_raw = torch.matmul(query, key_transposed)
            attn_weights = attn_weights_raw * scaling
            
            # 记录 QK^T matmul 的输入信息（不记录输出）
            hook_data[f'layer_{layer_id}_QK^T'] = {
                'operation': 'Q @ K^T',
                'input1_name': 'query',
                'input1_shape': list(query.shape),
                'input1_sparsity': float((query.numel() - query.count_nonzero().item()) / query.numel() if query.numel() > 0 else 0),
                'input2_name': 'key_transposed',
                'input2_shape': list(key_transposed.shape),
                'input2_sparsity': float((key_transposed.numel() - key_transposed.count_nonzero().item()) / key_transposed.numel() if key_transposed.numel() > 0 else 0),
                'scaling_factor': float(scaling),
            }
            
            # 保存输入tensor到内存缓存
            tensor_cache[f'layer_{layer_id}_QK^T_input1_query'] = query.detach().cpu().float()
            tensor_cache[f'layer_{layer_id}_QK^T_input2_key_transposed'] = key_transposed.detach().cpu().float()
            
            # 应用 attention mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # ===== 捕获 softmax 操作 =====
            attn_weights_softmax = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            try:
                top_1_percent_threshold = safe_quantile(attn_weights_softmax, 0.99)
                top_10_percent_threshold = safe_quantile(attn_weights_softmax, 0.90)
            except Exception:
                top_1_percent_threshold = None
                top_10_percent_threshold = None
            
            # 记录 softmax 后的 attention weights
            hook_data[f'layer_{layer_id}_attention_weights'] = {
                'operation': 'softmax(Q@K^T)',
                'input_shape': list(attn_weights.shape),
                'output_shape': list(attn_weights_softmax.shape),
                'output_numel': int(attn_weights_softmax.numel()),
                'output_sparsity': float((attn_weights_softmax.numel() - attn_weights_softmax.count_nonzero().item()) / attn_weights_softmax.numel() if attn_weights_softmax.numel() > 0 else 0),
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
            
            # ===== 捕获 softmax(QK^T) @ V matmul 的输入 =====
            attn_output = torch.matmul(attn_weights_dropped, value)
            
            # 记录 attention_weights @ V matmul 的输入信息（不记录输出）
            hook_data[f'layer_{layer_id}_attention_weights_matmul'] = {
                'operation': 'softmax(Q@K^T) @ V',
                'input1_name': 'attention_weights_softmax',
                'input1_shape': list(attn_weights_dropped.shape),
                'input1_sparsity': float((attn_weights_dropped.numel() - attn_weights_dropped.count_nonzero().item()) / attn_weights_dropped.numel() if attn_weights_dropped.numel() > 0 else 0),
                'input2_name': 'value',
                'input2_shape': list(value.shape),
                'input2_sparsity': float((value.numel() - value.count_nonzero().item()) / value.numel() if value.numel() > 0 else 0),
            }
            
            # 保存输入tensor到内存缓存
            tensor_cache[f'layer_{layer_id}_attention_weights_matmul_input1_attention_weights'] = attn_weights_dropped.detach().cpu().float()
            tensor_cache[f'layer_{layer_id}_attention_weights_matmul_input2_value'] = value.detach().cpu().float()
            
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
        self.tensor_cache = {}
        self.layer_counter = 0
    
    def save_tensors(self, output_dir):
        """保存所有缓存的tensor到指定目录的tensors子目录"""
        if not self.tensor_cache:
            print("⚠ 没有缓存的 attention tensor 数据")
            return
        
        tensor_dir = os.path.join(output_dir, 'tensors')
        os.makedirs(tensor_dir, exist_ok=True)
        
        saved_count = 0
        for tensor_name, tensor_data in self.tensor_cache.items():
            try:
                # 使用 .pt 格式保存
                tensor_file = os.path.join(tensor_dir, f"{tensor_name}.pt")
                torch.save(tensor_data, tensor_file)
                saved_count += 1
            except Exception as e:
                print(f"⚠ 保存 tensor {tensor_name} 失败: {e}")
        
        print(f"  ✓ 已保存 {saved_count} 个 attention matmul 输入 tensor 到 tensors/")
    
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
                print(f"  Q@K^T inputs: query_shape={qkt['input1_shape']}, "
                      f"key_transposed_shape={qkt['input2_shape']}")
            
            # Attention weights
            attn_key = f'layer_{layer_id}_attention_weights'
            if attn_key in self.data:
                attn = self.data[attn_key]
                print(f"  Attention weights: shape={attn['output_shape']}, "
                      f"sparsity={attn['output_sparsity']:.4f}")
            
            # Attention weights @ V matmul
            attn_v_key = f'layer_{layer_id}_attention_weights_matmul'
            if attn_v_key in self.data:
                attn_v = self.data[attn_v_key]
                print(f"  Attn@V inputs: attention_weights_shape={attn_v['input1_shape']}, "
                      f"value_shape={attn_v['input2_shape']}")
            
            print()


if __name__ == "__main__":
    print("OPTAttentionHook 已准备就绪")
    print("使用示例见 test_opt_attention_hook.py")
