"""
改进的 GEMM 操作 Hook 模块，用于更细致地捕获 Attention 中的 Q/K/V/QK^T/Attention@V 操作
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime


class ImprovedGEMMHookManager:
    """改进的 GEMM Hook 管理器，更好地支持 Attention 操作的捕获"""
    
    def __init__(self, output_dir, model_name, config_name, dataset_name):
        """
        初始化改进的 GEMM Hook 管理器
        
        Args:
            output_dir: 输出目录
            model_name: 模型名称
            config_name: 配置名称（如 dense, gmp, sparsity0.5等）
            dataset_name: 数据集名称（wikitext2, ptb, c4）
        """
        self.output_dir = output_dir
        self.model_name = model_name
        self.config_name = config_name
        self.dataset_name = dataset_name
        
        # 创建基础输出目录
        self.base_dir = os.path.join(
            output_dir, 
            f"{model_name}_{config_name}_{dataset_name}"
        )
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.inference_count = 0
        self.current_inference_dir = None
        self.hooks = []
        self.gemm_data = {}
        self.tensor_cache = {}
        
        # Attention 中间结果缓存
        self.attention_intermediates = {}
        
    def create_inference_dir(self):
        """为每次推理创建新目录"""
        self.inference_count += 1
        self.current_inference_dir = os.path.join(
            self.base_dir,
            f"inference_{self.inference_count:03d}"
        )
        os.makedirs(self.current_inference_dir, exist_ok=True)
        self.gemm_data = {}
        self.tensor_cache = {}
        self.attention_intermediates = {}
        
    def register_hooks(self, model):
        """注册所有线性层的 hook，只记录 input 和 weight"""
        self.hooks = []
        
        # 首先收集所有Linear层及其索引
        linear_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_modules.append((name, module))
        
        # 在闭包外保存对self的引用
        manager_ref = self
        
        # 为每个Linear层注册hook
        for layer_idx, (module_name, module) in enumerate(linear_modules):
            def linear_hook(module, input, output, module_name=module_name, idx=layer_idx):
                if len(input) >= 1:
                    inp_tensor = input[0].detach().cpu().float()
                    weight_tensor = module.weight.data.detach().cpu().float()
                    
                    # 使用 layer_数字 作为开头，保持命名一致
                    gemm_key_input = f"layer_{idx}_linear_input_{module_name}"
                    gemm_key_weight = f"layer_{idx}_linear_weight_{module_name}"
                    
                    # 记录统计信息到manager_ref
                    manager_ref.gemm_data[gemm_key_input] = {
                        'operation': 'linear_input',
                        'input_shape': tuple(inp_tensor.shape),
                        'input_sparsity': manager_ref._compute_sparsity(inp_tensor),
                        'input_format': 'float32',
                    }
                    
                    manager_ref.gemm_data[gemm_key_weight] = {
                        'operation': 'linear_weight',
                        'weight_shape': tuple(weight_tensor.shape),
                        'weight_sparsity': manager_ref._compute_sparsity(weight_tensor),
                        'weight_format': 'float32',
                    }
                    
                    # 保存tensor到缓存
                    manager_ref.tensor_cache[gemm_key_input] = inp_tensor
                    manager_ref.tensor_cache[gemm_key_weight] = weight_tensor
            
            hook = module.register_forward_hook(linear_hook)
            self.hooks.append(hook)
        
        # 调试信息：打印注册的hooks数量
        print(f"✓ 已注册 {len(self.hooks)} 个 Linear 层 hook")
        
    def _compute_sparsity(self, tensor):
        """计算张量的稀疏性"""
        if tensor is None or tensor.numel() == 0:
            return 0.0
        return float((tensor == 0).sum().item() / tensor.numel())
    
    def _restore_matmul(self):
        """恢复原始的 matmul 函数（已弃用，保留向后兼容）"""
        pass
    
    def remove_hooks(self):
        """移除所有注册的 hook"""
        for hook in self.hooks:
            hook.remove()
        self._restore_matmul()
        self.hooks = []
    
    def save_inference_data(self, ppl_result):
        """保存当前推理的数据"""
        if self.current_inference_dir is None:
            return
        
        # 保存 GEMM 数据统计
        stats_file = os.path.join(self.current_inference_dir, 'gemm_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.gemm_data, f, indent=2)
        
        print(f"  ✓ 已保存 {len(self.gemm_data)} 个 linear 层统计信息到 gemm_stats.json")
        
        # 保存缓存的 tensor 数据到 tensors 目录（统一存放所有tensor）
        if self.tensor_cache:
            tensor_dir = os.path.join(self.current_inference_dir, 'tensors')
            os.makedirs(tensor_dir, exist_ok=True)
            
            saved_count = 0
            for tensor_name, tensor_data in self.tensor_cache.items():
                tensor_file = os.path.join(tensor_dir, f"{tensor_name}.pt")
                try:
                    torch.save(tensor_data, tensor_file)
                    saved_count += 1
                except Exception as e:
                    print(f"⚠ 保存 tensor {tensor_name} 失败: {e}")
            
            print(f"  ✓ 已保存 {saved_count} 个 linear 层 tensor 到 tensors/")
        
        # 保存元数据
        metadata = {
            'model': self.model_name,
            'config': self.config_name,
            'dataset': self.dataset_name,
            'ppl': float(ppl_result),
            'timestamp': datetime.now().isoformat(),
            'inference_number': self.inference_count,
            'total_operations': len(self.gemm_data),
        }
        
        meta_file = os.path.join(self.current_inference_dir, 'metadata.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"已保存第 {self.inference_count} 次推理数据到: {self.current_inference_dir}")
        print(f"  - GEMM 操作统计: {len(self.gemm_data)} 项")
        print(f"  - 缓存的 Tensor: {len(self.tensor_cache)} 个")
    
    def save_summary(self, ppl_results_list):
        """保存评估摘要"""
        summary_file = os.path.join(self.base_dir, 'evaluation_summary.json')
        summary = {
            'model': self.model_name,
            'config': self.config_name,
            'dataset': self.dataset_name,
            'ppl_results': [float(p) for p in ppl_results_list],
            'average_ppl': float(np.mean(ppl_results_list)),
            'min_ppl': float(np.min(ppl_results_list)),
            'max_ppl': float(np.max(ppl_results_list)),
            'total_inferences': self.inference_count,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"已保存评估摘要到: {summary_file}")
