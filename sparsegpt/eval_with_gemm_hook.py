"""
使用 GEMM Hook 进行模型评估，记录所有推理的 GEMM 输入和稀疏性
运行命令示例：
python eval_with_gemm_hook.py /home/zhaojun/proj26/models/opt-125m --data-root /home/zhaojun/proj26/datasets --num-inferences 5
"""
import os
import time
import json
import torch
import torch.nn as nn
import argparse
import numpy as np

from datautils import get_loaders
from modelutils import find_layers, DEV
#from gemm_hook import GEMMHookManager
from gemm_hook_improved import ImprovedGEMMHookManager as GEMMHookManager
from opt_attention_hook import OPTAttentionHook

try:
    from opt import get_opt
except:
    # Fallback: 在本文件中定义 get_opt
    def get_opt(model):
        import torch
        def skip(*args, **kwargs):
            pass
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(
            model,
            torch_dtype='auto',
            attn_implementation='eager',
        )
        model.seqlen = model.config.max_position_embeddings
        return model


def _ensure_eager_attention(model):
    """强制使用 eager attention，确保可捕获 QK^T 和 attention@V matmul。"""
    if hasattr(model, 'config'):
        model.config._attn_implementation = 'eager'


def evaluate_with_gemm_hook(model, dataloader, testloader, hook_manager, dataset_name, inference_idx=0, log_wandb=False):
    """
    使用 GEMM hook 进行评估，支持 hook 功能
    
    Args:
        inference_idx: 推理索引，用于在多次推理中使用不同的数据片段
    """
    model.eval()
    _ensure_eager_attention(model)
    
    # 确保模型在正确的设备上
    model = model.to(DEV)
    
    # 注册 hook
    hook_manager.register_hooks(model)
    attention_hook = OPTAttentionHook(save_tensors=False)
    attention_hook.install_hook()
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    nsamples = testloader.input_ids.shape[1] // model.seqlen if hasattr(testloader, 'input_ids') else 32
    testenc = testloader.input_ids if hasattr(testloader, 'input_ids') else testloader
    
    if nsamples == 0:
        print(f"警告: nsamples 为 0，设置为 1")
        nsamples = 1
    
    nlls = []
    
    # 使用不同的起始位置以获得不同的数据片段
    # 这样每次推理都会评估不同的数据
    max_offset = testenc.shape[1] - (5 * model.seqlen)
    if max_offset > 0:
        offset = (inference_idx * model.seqlen) % max_offset
    else:
        offset = 0
    
    try:
        with torch.no_grad():
            for i in range(min(nsamples, 5)):  # 最多计算 5 个样本
                start_idx = offset + (i * model.seqlen)
                end_idx = start_idx + model.seqlen
                
                # 确保不超出范围
                if end_idx > testenc.shape[1]:
                    break
                
                batch = testenc[:, start_idx:end_idx].to(DEV)
                
                # 前向传播收集 GEMM 数据
                outputs = model(batch)
                lm_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:]
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood.item())
    finally:
        attention_hook.remove_hook()
    
    if not nlls:
        raise ValueError("评估失败：没有计算任何样本，nlls 列表为空")
    
    ppl = torch.exp(torch.tensor(sum(nlls) / (len(nlls) * model.seqlen)))

    # 保存 attention matmul 统计（仅当本次推理目录已创建）
    if hook_manager.current_inference_dir is not None:
        attention_stats_path = os.path.join(hook_manager.current_inference_dir, 'attention_matmul_stats.json')
        with open(attention_stats_path, 'w') as f:
            json.dump(attention_hook.get_summary(), f, indent=2)
    
    # 移除 hook
    hook_manager.remove_hooks()
    
    model.config.use_cache = use_cache
    
    return float(ppl.item())


def get_config_name(args):
    """根据参数生成配置名称"""
    config_parts = []
    
    if args.gmp:
        config_parts.append("gmp")
    elif args.sparsity > 0:
        config_parts.append(f"sp{int(args.sparsity * 100)}")
    elif args.prunen > 0 and args.prunem > 0:
        config_parts.append(f"{args.prunen}:{args.prunem}")
    else:
        config_parts.append("dense")
    
    if args.wbits < 16:
        config_parts.append(f"q{args.wbits}")
    
    return "_".join(config_parts) if config_parts else "dense"


def run_evaluation_with_configs(model_path, num_inferences=5, data_root=None, output_dir=None):
    """
    运行多个配置的评估
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取模型名称
    model_name = os.path.basename(model_path).replace('/', '_')
    
    # 定义 5 种配置
    configs = [
        {'gmp': False, 'sparsity': 0, 'prunen': 0, 'prunem': 0, 'wbits': 16, 'name': 'dense'},
        {'gmp': True, 'sparsity': 0.5, 'prunen': 0, 'prunem': 0, 'wbits': 16, 'name': 'gmp'},
        {'gmp': False, 'sparsity': 0.5, 'prunen': 0, 'prunem': 0, 'wbits': 16, 'name': 'sp50'},
        {'gmp': False, 'sparsity': 0, 'prunen': 2, 'prunem': 4, 'wbits': 16, 'name': '2:4'},
        {'gmp': False, 'sparsity': 0.5, 'prunen': 0, 'prunem': 0, 'wbits': 4, 'name': 'sp50_q4'},
    ]
    
    datasets = ['wikitext2', 'ptb', 'c4']
    
    # 加载模型一次
    print(f"加载模型: {model_path}")
    model = get_opt(model_path)
    _ensure_eager_attention(model)
    model = model.to(DEV)
    model.eval()
    
    # 对每个配置和数据集进行评估
    for config in configs:
        config_name = config.pop('name')
        
        # 配置参数
        gmp = config['gmp']
        sparsity = config['sparsity']
        prunen = config['prunen']
        prunem = config['prunem']
        wbits = config['wbits']
        
        for dataset in datasets:
            print(f"\n{'='*70}")
            print(f"配置: {config_name:15s} | 数据集: {dataset:10s}")
            print(f"{'='*70}")
            
            # 加载数据集
            print(f"  加载数据集...", end=' ')
            dataloader, testloader = get_loaders(
                dataset, nsamples=128, seed=0, model=model_path, 
                seqlen=model.seqlen, data_root=data_root
            )
            print("完成")
            
            # 创建 hook 管理器
            hook_manager = GEMMHookManager(
                output_dir, model_name, config_name, dataset
            )
            
            ppl_results = []
            
            # 运行多次推理
            for inference_idx in range(num_inferences):
                print(f"  推理 [{inference_idx + 1:2d}/{num_inferences}]", end='')
                
                # 仅在非 dense 配置时创建推理目录和保存数据
                if config_name != 'dense':
                    hook_manager.create_inference_dir()
                
                # 应用配置（如果需要剪枝）
                if gmp and sparsity > 0:
                    print(" (应用 GMP 剪枝)", end='')
                    layers = model.model.decoder.layers
                    for layer in layers:
                        subset = find_layers(layer)
                        for name, module in subset.items():
                            W = module.weight.data
                            thresh = torch.sort(torch.abs(W.flatten()))[0][
                                int(W.numel() * sparsity)
                            ]
                            W.data[torch.abs(W.data) <= thresh] = 0
                elif sparsity > 0 and prunen == 0:
                    print(" (应用 SparseGPT 剪枝)", end='')
                    layers = model.model.decoder.layers
                    for layer in layers:
                        subset = find_layers(layer)
                        for name, module in subset.items():
                            W = module.weight.data
                            thresh = torch.sort(torch.abs(W.flatten()))[0][
                                int(W.numel() * sparsity)
                            ]
                            W.data[torch.abs(W.data) <= thresh] = 0
                
                # 进行评估
                ppl = evaluate_with_gemm_hook(
                    model, dataloader, testloader, hook_manager, dataset, inference_idx=inference_idx
                )
                ppl_results.append(ppl)
                print("")  # 换行
                
                # 仅在非 dense 配置时保存数据
                if config_name != 'dense':
                    hook_manager.save_inference_data(ppl)
                
                # 重新加载模型以清除任何修改
                if inference_idx < num_inferences - 1:  # 最后一次不需要重新加载
                    model = get_opt(model_path)
                    _ensure_eager_attention(model)
                    model = model.to(DEV)
                    model.eval()
            
            # 输出推理结果统计
            avg_ppl = np.mean(ppl_results)
            print(f"  平均 PPL: {avg_ppl:.4f} (最小: {np.min(ppl_results):.4f}, 最大: {np.max(ppl_results):.4f})")
            
            # 保存评估摘要
            hook_manager.save_summary(ppl_results)
            
            # 重新加载模型进行下一个数据集
            model = get_opt(model_path)
            _ensure_eager_attention(model)
            model = model.to(DEV)
            model.eval()
    
    print(f"\n\n{'='*70}")
    print(f"✓ 所有评估完成！输出目录: {output_dir}")
    print(f"{'='*70}")
    print(f"\n目录结构:")
    print(f"  {output_dir}/")
    print(f"    ├── model_config_dataset/")
    print(f"    │  ├── evaluation_summary.json")
    print(f"    │  ├── inference_001/")
    print(f"    │  │  ├── gemm_stats.json")
    print(f"    │  │  └── metadata.json")
    print(f"    │  ├── inference_002/")
    print(f"    │  └── ...inference_010/")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用 GEMM Hook 进行模型评估')
    
    parser.add_argument('model', type=str, help='模型路径或 HuggingFace ID')
    parser.add_argument('--num-inferences', type=int, default=5, 
                        help='每个配置运行的推理次数')
    parser.add_argument('--data-root', type=str, default='/home/zhaojun/proj26/datasets',
                        help='本地数据集根目录')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认为当前目录的 output 文件夹）')
    
    args = parser.parse_args()
    
    run_evaluation_with_configs(
        args.model,
        num_inferences=args.num_inferences,
        data_root=args.data_root,
        output_dir=args.output_dir or os.path.join(os.path.dirname(__file__), 'output')
    )
