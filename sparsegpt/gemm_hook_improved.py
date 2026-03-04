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
        """注册所有线性层和 attention matmul 的 hook"""
        self.hooks = []
        gemm_counter = [0]
        
        for name, module in model.named_modules():
            # Hook for Linear layers
            if isinstance(module, torch.nn.Linear):
                def linear_hook(module, input, output, module_name=name, idx=gemm_counter[0]):
                    if len(input) >= 1:
                        inp_tensor = input[0].detach().cpu()
                        weight_tensor = module.weight.data.detach().cpu()
                        
                        gemm_key = f"linear_{idx}_{module_name}"
                        self.gemm_data[gemm_key] = {
                            'input_shape': tuple(inp_tensor.shape),
                            'weight_shape': tuple(weight_tensor.shape),
                            'input_format': 'float32',
                            'input_sparsity': self._compute_sparsity(inp_tensor),
                            'weight_sparsity': self._compute_sparsity(weight_tensor),
                        }
                
                hook = module.register_forward_hook(linear_hook)
                self.hooks.append(hook)
                gemm_counter[0] += 1
        
        # 为 attention 层添加更详细的 hook
        self._register_detailed_attention_hooks(model)
        
    def _register_detailed_attention_hooks(self, model):
        """
        为 Attention 层添加详细的 hook，捕获：
        1. Q/K/V 的投影输出
        2. Q @ K^T 的 matmul
        3. softmax(Q@K^T) @ V 的 matmul
        """
        attn_counter = [0]
        
        for name, module in model.named_modules():
            if 'self_attn' in name and not 'layer_norm' in name:
                block_id = attn_counter[0]
                
                # 为子模块（q_proj, k_proj, v_proj, out_proj）注册 hook
                if hasattr(module, 'q_proj'):
                    self._hook_projection(module.q_proj, f"block_{block_id}_q_proj")
                if hasattr(module, 'k_proj'):
                    self._hook_projection(module.k_proj, f"block_{block_id}_k_proj")
                if hasattr(module, 'v_proj'):
                    self._hook_projection(module.v_proj, f"block_{block_id}_v_proj")
                if hasattr(module, 'out_proj'):
                    self._hook_projection(module.out_proj, f"block_{block_id}_out_proj")
                
                # 为 attention 整体的前向传播注册 hook 以捕获 QKV matmul
                def attention_forward_hook(mod, input, output, bid=block_id, aname=name):
                    """Wrapper hook 来捕获 attention 内部的 matmul"""
                    try:
                        if isinstance(output, tuple):
                            attn_output = output[0]
                        else:
                            attn_output = output
                            
                        if isinstance(attn_output, torch.Tensor):
                            attn_key = f"attention_output_block_{bid}"
                            self.gemm_data[attn_key] = {
                                'operation': 'attention_output',
                                'module': aname,
                                'output_shape': tuple(attn_output.shape),
                                'output_numel': int(attn_output.numel()),
                                'output_sparsity': float(self._compute_sparsity(
                                    attn_output.detach().cpu().float()
                                )),
                            }
                            self.tensor_cache[attn_key] = attn_output.detach().cpu().float()
                    except Exception as e:
                        pass
                
                hook = module.register_forward_hook(attention_forward_hook)
                self.hooks.append(hook)
                attn_counter[0] += 1
        
        # 现在进行更激进的 matmul hook 来捕获 QK^T 和 attention*V
        self._register_attention_matmul_hooks(model)
    
    def _hook_projection(self, linear_module, proj_name):
        """为投影层（q/k/v/out_proj）添加 hook"""
        def proj_hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    out_cpu = output.detach().cpu().float()
                    self.gemm_data[f"{proj_name}_output"] = {
                        'operation': 'projection_output',
                        'projection': proj_name,
                        'output_shape': tuple(out_cpu.shape),
                        'output_numel': int(out_cpu.numel()),
                        'output_sparsity': float(self._compute_sparsity(out_cpu)),
                    }
                    self.tensor_cache[f"{proj_name}_output"] = out_cpu
                    
                    # 缓存 Q/K/V 供后续计算使用
                    self.attention_intermediates[proj_name] = out_cpu
            except Exception as e:
                pass
        
        hook = linear_module.register_forward_hook(proj_hook)
        self.hooks.append(hook)
    
    def _register_attention_matmul_hooks(self, model):
        """
        为 attention 中的 matmul 操作添加 hook
        这包括：
        - Q @ K^T (attention scores)
        - softmax(scores) @ V (attention output)
        """
        # 保存原始的 matmul 函数
        self._original_matmul = torch.matmul
        self._original_bmm = torch.bmm
        self._original_mm = torch.mm
        
        matmul_counter = [0]
        layer_matmul_counter = {}  # 追踪每个 layer 中的 matmul 数量
        
        def create_attention_aware_matmul(op_name, original_op):
            """创建能够识别 attention matmul 的 hook"""
            def hooked_matmul(input1, input2, *args, **kwargs):
                result = original_op(input1, input2, *args, **kwargs)
                
                try:
                    # 检查是否是 attention 相关的 matmul
                    if isinstance(input1, torch.Tensor) and isinstance(input2, torch.Tensor):
                        # 获取形状用于识别
                        shape1 = input1.shape
                        shape2 = input2.shape
                        result_shape = result.shape
                        
                        # 启发式判断：
                        # QK^T: shape1 = [..., seq_len, head_dim], shape2 = [..., head_dim, seq_len]
                        #       result = [..., seq_len, seq_len]
                        # attn*V: shape1 = [..., seq_len, seq_len], shape2 = [..., seq_len, head_dim]
                        #       result = [..., seq_len, head_dim]
                        
                        is_qkt_like = (len(shape1) == len(shape2) and 
                                     len(shape1) >= 2 and
                                     shape1[-1] == shape2[-2] and
                                     shape1[-2] == result_shape[-2])
                        
                        matmul_counter[0] += 1
                        matmul_idx = matmul_counter[0]
                        
                        inp1_cpu = input1.detach().cpu().float() if input1.numel() > 100 else None
                        inp2_cpu = input2.detach().cpu().float() if input2.numel() > 100 else None
                        result_cpu = result.detach().cpu().float()
                        
                        matmul_key = f"matmul_{matmul_idx}_{op_name}"
                        self.gemm_data[matmul_key] = {
                            'operation': op_name,
                            'input1_shape': tuple(shape1),
                            'input2_shape': tuple(shape2),
                            'output_shape': tuple(result_shape),
                            'is_attention_like': is_qkt_like,
                            'input1_sparsity': float(self._compute_sparsity(inp1_cpu)) if inp1_cpu is not None else 0,
                            'input2_sparsity': float(self._compute_sparsity(inp2_cpu)) if inp2_cpu is not None else 0,
                            'output_sparsity': float(self._compute_sparsity(result_cpu)),
                        }
                        
                        # 保存较小的 tensor
                        if inp1_cpu is not None and inp1_cpu.numel() < 1e7:
                            self.tensor_cache[f"{matmul_key}_input1"] = inp1_cpu
                        if inp2_cpu is not None and inp2_cpu.numel() < 1e7:
                            self.tensor_cache[f"{matmul_key}_input2"] = inp2_cpu
                        if result_cpu.numel() < 1e7:
                            self.tensor_cache[f"{matmul_key}_output"] = result_cpu
                            
                except Exception as e:
                    pass
                
                return result
            
            return hooked_matmul
        
        # 替换全局的 matmul 函数
        torch.matmul = create_attention_aware_matmul('matmul', self._original_matmul)
        torch.bmm = create_attention_aware_matmul('bmm', self._original_bmm)
        torch.mm = create_attention_aware_matmul('mm', self._original_mm)
    
    def _restore_matmul(self):
        """恢复原始的 matmul 函数"""
        if hasattr(self, '_original_matmul'):
            torch.matmul = self._original_matmul
        if hasattr(self, '_original_bmm'):
            torch.bmm = self._original_bmm
        if hasattr(self, '_original_mm'):
            torch.mm = self._original_mm
    
    def _compute_sparsity(self, tensor):
        """计算张量的稀疏性"""
        if tensor is None or tensor.numel() == 0:
            return 0.0
        return float((tensor == 0).sum().item() / tensor.numel())
    
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
        
        # 保存缓存的 tensor 数据到磁盘
        if self.tensor_cache:
            tensor_dir = os.path.join(self.current_inference_dir, 'tensors')
            os.makedirs(tensor_dir, exist_ok=True)
            
            for tensor_name, tensor_data in self.tensor_cache.items():
                tensor_file = os.path.join(tensor_dir, f"{tensor_name}.pt")
                try:
                    torch.save(tensor_data, tensor_file)
                except:
                    pass
        
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
