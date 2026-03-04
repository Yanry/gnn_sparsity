"""
GEMM 操作 Hook 模块，用于在评估时捕获所有线性层和 matmul 操作的输入
"""
import os
import json
import torch
import numpy as np
from datetime import datetime


class GEMMHookManager:
    """管理 GEMM 操作的 hook"""
    
    def __init__(self, output_dir, model_name, config_name, dataset_name):
        """
        初始化 GEMM Hook 管理器
        
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
        self.tensor_cache = {}  # 初始化 tensor 缓存
        
    def create_inference_dir(self):
        """为每次推理创建新目录"""
        self.inference_count += 1
        self.current_inference_dir = os.path.join(
            self.base_dir,
            f"inference_{self.inference_count:03d}"
        )
        os.makedirs(self.current_inference_dir, exist_ok=True)
        self.gemm_data = {}
        self.tensor_cache = {}  # 清空 tensor 缓存
        
    def register_hooks(self, model):
        """注册所有线性层和 matmul 的 hook"""
        self.hooks = []
        gemm_counter = [0]  # 使用列表来在闭包中修改计数器
        
        for name, module in model.named_modules():
            # Hook for Linear layers
            if isinstance(module, torch.nn.Linear):
                def linear_hook(module, input, output, module_name=name, idx=gemm_counter[0]):
                    if len(input) >= 1:
                        # input[0] 是输入张量，module.weight 是权重
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
                        
                        # 保存实际的 tensor 数据
                        if self.current_inference_dir is not None:
                            tensor_dir = os.path.join(self.current_inference_dir, 'tensors')
                            os.makedirs(tensor_dir, exist_ok=True)
                            
                            # 保存输入和权重 tensor
                            inp_file = os.path.join(tensor_dir, f"{gemm_key}_input.pt")
                            weight_file = os.path.join(tensor_dir, f"{gemm_key}_weight.pt")
                            
                            torch.save(inp_tensor, inp_file)
                            torch.save(weight_tensor, weight_file)
                
                hook = module.register_forward_hook(linear_hook)
                self.hooks.append(hook)
                gemm_counter[0] += 1
        
        # Hook for matmul operations (includes attention qk^T and attention*V)
        self._register_matmul_hooks(model)
        
        # 额外：为 attention 层添加 hook 以捕获 attention scores
        self._register_attention_hooks(model)
        
    def _compute_sparsity(self, tensor):
        """计算张量的稀疏性"""
        if tensor.numel() == 0:
            return 0.0
        return float((tensor == 0).sum().item() / tensor.numel())
    
    def _register_matmul_hooks(self, model):
        """为 matmul 操作注册 hook 以捕获 attention 计算"""
        self.matmul_counter = [0]
        self._matmul_threshold = 0  # 捕获所有 matmul 操作
        
        # 保存原始函数
        self._original_matmul = torch.matmul
        self._original_bmm = torch.bmm
        self._original_mm = torch.mm
        self._original_tensor_matmul = torch.Tensor.__matmul__
        
        # 创建闭包以访问 self
        def create_matmul_hook(op_name, original_func):
            """创建一个通用的 matmul hook"""
            def hook(input1, input2):
                result = original_func(input1, input2)
                
                # 记录 matmul 操作的信息
                try:
                    inp1_cpu = input1.detach().cpu().float() if isinstance(input1, torch.Tensor) else input1
                    inp2_cpu = input2.detach().cpu().float() if isinstance(input2, torch.Tensor) else input2
                    result_cpu = result.detach().cpu().float() if isinstance(result, torch.Tensor) else result
                    
                    # 记录足够大的矩阵操作
                    if isinstance(inp1_cpu, torch.Tensor) and isinstance(inp2_cpu, torch.Tensor):
                        if inp1_cpu.numel() > self._matmul_threshold:
                            matmul_key = f"{op_name}_{self.matmul_counter[0]}"
                            self.matmul_counter[0] += 1
                            
                            self.gemm_data[matmul_key] = {
                                'operation': op_name,
                                'input1_shape': tuple(inp1_cpu.shape),
                                'input2_shape': tuple(inp2_cpu.shape),
                                'output_shape': tuple(result_cpu.shape),
                                'input_format': 'float32',
                                'input1_numel': int(inp1_cpu.numel()),
                                'input2_numel': int(inp2_cpu.numel()),
                                'input1_sparsity': float(self._compute_sparsity(inp1_cpu)),
                                'input2_sparsity': float(self._compute_sparsity(inp2_cpu)),
                            }
                            
                            # 缓存 tensor 数据到内存，稍后统一保存
                            self.tensor_cache[f"{matmul_key}_input1"] = inp1_cpu
                            self.tensor_cache[f"{matmul_key}_input2"] = inp2_cpu
                            self.tensor_cache[f"{matmul_key}_output"] = result_cpu
                except Exception as e:
                    # 忽略保存失败的情况
                    import traceback
                    pass
                
                return result
            return hook
        
        # 替换各种矩阵乘法函数
        torch.matmul = create_matmul_hook('matmul', self._original_matmul)
        torch.bmm = create_matmul_hook('bmm', self._original_bmm)
        torch.mm = create_matmul_hook('mm', self._original_mm)
        torch.Tensor.__matmul__ = create_matmul_hook('tensor_matmul', self._original_tensor_matmul)
    
    def _register_attention_hooks(self, model):
        """为 attention 模块注册 hook 以直接捕获 attention 操作"""
        attention_counter = [0]
        
        # 查找所有 attention 层（OPT 使用 OPTAttention）
        for name, module in model.named_modules():
            if 'self_attn' in name or 'attention' in name.lower():
                # 为 attention 层的前向输出注册 hook
                def attention_hook(module, input, output, idx=attention_counter[0], mod_name=name):
                    """Hook 用于捕获 attention 的输出"""
                    try:
                        # output 通常是 (attn_output, attn_weights, ...)
                        if isinstance(output, tuple):
                            attn_output = output[0]  # attention 输出
                        else:
                            attn_output = output
                        
                        if isinstance(attn_output, torch.Tensor) and attn_output.numel() > 100:
                            attn_key = f"attn_output_{idx}"
                            self.gemm_data[attn_key] = {
                                'operation': 'attention_output',
                                'module': mod_name,
                                'output_shape': tuple(attn_output.shape),
                                'output_numel': int(attn_output.numel()),
                                'output_sparsity': float(self._compute_sparsity(attn_output.detach().cpu().float())),
                            }
                            
                            # 缓存 tensor
                            self.tensor_cache[attn_key] = attn_output.detach().cpu().float()
                    except Exception as e:
                        pass
                
                hook = module.register_forward_hook(attention_hook)
                self.hooks.append(hook)
                attention_counter[0] += 1
    
    def _restore_matmul(self):
        """恢复原始的 matmul 相关函数"""
        if hasattr(self, '_original_matmul'):
            torch.matmul = self._original_matmul
        if hasattr(self, '_original_bmm'):
            torch.bmm = self._original_bmm
        if hasattr(self, '_original_mm'):
            torch.mm = self._original_mm
        if hasattr(self, '_original_tensor_matmul'):
            torch.Tensor.__matmul__ = self._original_tensor_matmul
    
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
                torch.save(tensor_data, tensor_file)
        
        # 保存 PPL 结果和元数据
        metadata = {
            'model': self.model_name,
            'config': self.config_name,
            'dataset': self.dataset_name,
            'ppl': float(ppl_result),
            'timestamp': datetime.now().isoformat(),
            'inference_number': self.inference_count,
            'total_gemm_operations': len(self.gemm_data),
        }
        
        meta_file = os.path.join(self.current_inference_dir, 'metadata.json')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 检查保存的 tensor 文件数量
        tensor_dir = os.path.join(self.current_inference_dir, 'tensors')
        tensor_count = 0
        if os.path.exists(tensor_dir):
            tensor_count = len([f for f in os.listdir(tensor_dir) if f.endswith('.pt')])
        
        print(f"已保存第 {self.inference_count} 次推理数据到: {self.current_inference_dir}")
        print(f"  - GEMM 统计: {len(self.gemm_data)} 个操作")
        if tensor_count > 0:
            print(f"  - Tensor 数据: {tensor_count} 个文件")
    
    def remove_hooks(self):
        """移除所有注册的 hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        # 恢复原始的 matmul 函数
        self._restore_matmul()
    
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


def register_matmul_hook(module, hook_manager):
    """
    为 F.linear 和各种 matmul 运算注册全局 hook
    这需要用函数级别的 hook，因为 matmul 不是 nn.Module
    """
    original_linear = torch.nn.functional.linear
    original_matmul = torch.matmul
    original_bmm = torch.bmm
    
    matmul_counter = [0]
    
    def hooked_linear(input, weight, bias=None):
        result = original_linear(input, weight, bias)
        sparsity = hook_manager._compute_sparsity(input.detach().cpu()) if hook_manager else 0.0
        return result
    
    def hooked_matmul(input, other):
        result = original_matmul(input, other)
        return result
    
    def hooked_bmm(input, mat2):
        result = original_bmm(input, mat2)
        return result
    
    # 替换函数
    torch.nn.functional.linear = hooked_linear
    torch.matmul = hooked_matmul
    torch.bmm = hooked_bmm
    
    return {
        'linear': original_linear,
        'matmul': original_matmul,
        'bmm': original_bmm,
    }


def restore_original_functions(original_funcs):
    """恢复原始的函数"""
    torch.nn.functional.linear = original_funcs['linear']
    torch.matmul = original_funcs['matmul']
    torch.bmm = original_funcs['bmm']
