"""
使用与 opt.py 完全相同的评估逻辑进行模型评估，同时收集 attention hook 数据

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
from sparsegpt import *
from datautils import get_loaders
from modelutils import find_layers, DEV
from gemm_hook_improved import ImprovedGEMMHookManager as GEMMHookManager
from opt_attention_hook import OPTAttentionHook

def get_after_decoder_layers_simple(s):
    if "decoder.layers." in s:
        return s.split("decoder.layers.", 1)[1]
    return None

def find_tensors_dirs(root_dir="output"):
    tensors_paths = []
    
    if not os.path.exists(root_dir):
        print(f"警告: 目录 '{root_dir}' 不存在")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == "tensors":
            tensors_paths.append(dirpath)
    
    return tensors_paths

# 导入 opt.py 中的剪枝实现
try:
    from opt import *
except ImportError as e:
    print(f"警告: 无法导入必要模块: {e}")
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
    if hasattr(model, 'config'):
        model.config._attn_implementation = 'eager'


@torch.no_grad()
def apply_sparsegpt_pruning(model, dataloader, dev, sparsity=0, prunen=0, prunem=0, 
                            wbits=16, nsamples=128, percdamp=0.01, blocksize=128):
    print('开始 SparseGPT 剪枝 ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # 只提取指定数量的样本
            if cache['i'] < nsamples:
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
            # 始终抛出异常以停止前向传播
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
        # 检查是否已收集足够的样本
        if cache['i'] >= nsamples:
            break
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('准备就绪，开始逐层剪枝 ...')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
            if wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(f'  层 {i}, 模块 {name}: 剪枝中 ...')
            gpts[name].fasterprune(
                sparsity, prunen=prunen, prunem=prunem, percdamp=percdamp, blocksize=blocksize
            )
            gpts[name].free()

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    # 显式删除大张量以释放显存
    del inps, outs, cache
    torch.cuda.empty_cache()
    
    model.config.use_cache = use_cache
    print('SparseGPT 剪枝完成。')


def evaluate_with_gemm_hook(model, dataloader, testloader, hook_manager, dataset_name, 
                           inference_idx=0, log_wandb=False, gmp=False, sparsity=0):
    """
    使用与 opt.py 完全相同的评估流程，同时收集 attention hook 数据
    类似于 opt.py 中的 opt_eval 函数实现
    
    改进：
    1. 采用逐层处理的方式（每次只有一层在 GPU 上）
    2. 收集 attention hook 数据（已优化内存使用）
    3. 准确的 PPL 计算
    
    Args:
        inference_idx: 推理索引，用于在多次推理中使用不同的数据片段
        gmp: 是否使用 GMP（全局幅度剪枝）基准
        sparsity: 如果使用 gmp，应用的稀疏度
    """
    print('开始评估 ...')
    
    model.eval()
    _ensure_eager_attention(model)
    
    # ===== 注册 Linear 层 hooks 以捕获 GEMM 操作 =====
    hook_manager.register_hooks(model)
    
    testenc = testloader.input_ids if hasattr(testloader, 'input_ids') else testloader
    # 使用更少的样本以节省内存（128 太大导致 OOM）
    max_eval_samples = 1
    nsamples = min(testenc.numel() // model.seqlen, max_eval_samples)
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    # 将嵌入层移到设备上
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(DEV)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(DEV)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(DEV) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(DEV) 
    layers[0] = layers[0].to(DEV)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=DEV
    )
    cache = {'i': 0, 'attention_mask': None}

    # 使用 Catcher 捕获输入
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(DEV)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    # 逐层处理（类似 opt_eval）
    for i in range(len(layers)):
        print(f"处理层 {i}")
        layer = layers[i].to(DEV)

        # 如果是 GMP，在这里应用剪枝（类似 opt.py 的 opt_eval）
        if gmp and sparsity > 0:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        # 前向传播
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        
        # 移到 CPU 并立即清理
        layers[i] = layer.cpu()
        del layer
        # 交换指针
        inps, outs = outs, inps
        # 清理内存
        torch.cuda.empty_cache()

    # 最后的层归一化和语言模型头
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(DEV)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(DEV)
    model.lm_head = model.lm_head.to(DEV)

    testenc = testenc.to(DEV)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"困惑度: {ppl.item():.3f}")
    
    # 显式删除大张量以释放显存
    del inps, outs, nlls, cache, attention_mask, testenc
    torch.cuda.empty_cache()
    
    # ===== 移除 Linear 层 hooks =====
    hook_manager.remove_hooks()
    
    model.config.use_cache = use_cache
    
    return float(ppl.item())


def collect_hook_data(model, testloader, hook_manager):
    """
    单独收集 hook 数据，只在第一个样本上运行以节省内存
    
    Args:
        model: 剪枝后的模型
        testloader: 测试数据加载器
        hook_manager: Hook 管理器实例
    """
    print("  收集 Attention Hook 数据...", end=' ')
    
    model.eval()
    
    # 确保模型中的所有层都在 GPU 上
    model = model.to(DEV)
    
    testenc = testloader.input_ids if hasattr(testloader, 'input_ids') else testloader
    
    # 注册 hook (启用 tensor 保存以获取完整数据)
    attention_hook = OPTAttentionHook(save_tensors=True)
    
    try:
        # 安装 hook
        print("安装Hook...", end=' ')
        attention_hook.install_hook()
        
        # 只使用一个样本来收集 hook 数据
        print("执行前向传播...", end=' ')
        batch = testenc[:, 0:model.seqlen].to(DEV)
        with torch.no_grad():
            output = model(batch, output_attentions=False)
        torch.cuda.empty_cache()
        
        # 检查是否捕获了数据
        num_operations = len(attention_hook.data)
        num_tensors = len(attention_hook.tensor_cache)
        print(f"完成 (捕获 {num_operations} 个操作)")
        
    except Exception as e:
        print(f"失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 移除 hook
        attention_hook.remove_hook()
        # 清理批次数据
        if 'batch' in locals():
            del batch
        torch.cuda.empty_cache()
    
    # 保存 attention 统计
    if hook_manager.current_inference_dir is not None:
        attention_stats_path = os.path.join(hook_manager.current_inference_dir, 'attention_matmul_stats.json')
        print(f"  保存 Hook 数据到 {hook_manager.current_inference_dir}...", end=' ')
        with open(attention_stats_path, 'w') as f:
            json.dump(attention_hook.get_summary(), f, indent=2)
        
        # 保存 tensor 数据
        print(f"保存 {len(attention_hook.tensor_cache)} 个 Tensor")
        if len(attention_hook.tensor_cache) > 0:
            attention_hook.save_tensors(hook_manager.current_inference_dir)
    
    # 清理 hook 数据以释放内存
    del attention_hook
    torch.cuda.empty_cache()


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
    运行多个配置的评估，使用与 opt.py 相同的剪枝实现
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取模型名称
    model_name = os.path.basename(model_path).replace('/', '_')
    
    # 定义 4 种配置（对应 batch_run.sh 中的配置）
    configs = [
        {'gmp': True, 'sparsity': 0.5, 'prunen': 0, 'prunem': 0, 'wbits': 16, 'name': 'gmp'},
        {'gmp': False, 'sparsity': 0.5, 'prunen': 0, 'prunem': 0, 'wbits': 16, 'name': 'sp50'},
        {'gmp': False, 'sparsity': 0, 'prunen': 2, 'prunem': 4, 'wbits': 16, 'name': '2:4'},
        {'gmp': False, 'sparsity': 0.5, 'prunen': 0, 'prunem': 0, 'wbits': 4, 'name': 'sp50_q4'},
    ]
    
    datasets = ['wikitext2', 'ptb', 'c4']
    
    # 先加载一次模型以获取 seqlen
    print(f"加载模型以获取配置信息: {model_path}")
    temp_model = get_opt(model_path)
    model_seqlen = temp_model.seqlen
    del temp_model
    torch.cuda.empty_cache()
    print(f"  模型序列长度: {model_seqlen}")
    
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
                seqlen=model_seqlen, data_root=data_root
            )
            print("完成")
            
            # 创建 hook 管理器
            hook_manager = GEMMHookManager(
                output_dir, model_name, config_name, dataset
            )
            
            ppl_results = []
            
            # 运行多次推理
            for inference_idx in range(num_inferences):
                print(f"  推理 [{inference_idx + 1:2d}/{num_inferences}]", end=' ')
                
                # 每次推理都重新加载模型
                model = get_opt(model_path)
                _ensure_eager_attention(model)
                model = model.to(DEV)
                model.eval()
                
                # 创建推理目录
                hook_manager.create_inference_dir()
                
                # 应用剪枝（如果需要）
                if gmp:
                    # GMP: 不需要提前剪枝，在评估时逐层应用
                    print(f"(GMP 基准 {int(sparsity*100)}%)", end=' ')
                    pruning_applied = False
                elif (sparsity > 0 and prunen == 0) or (prunen > 0 and prunem > 0):
                    # SparseGPT 或 N:M 剪枝: 调用 apply_sparsegpt_pruning
                    if prunen > 0:
                        print(f"(SparseGPT {prunen}:{prunem} 剪枝)", end=' ')
                    else:
                        print(f"(SparseGPT {int(sparsity*100)}% 剪枝)", end=' ')
                    
                    if wbits < 16:
                        print(f"+ {wbits}-bit 量化)", end=' ')
                    
                    apply_sparsegpt_pruning(
                        model, dataloader, DEV, 
                        sparsity=sparsity, 
                        prunen=prunen, 
                        prunem=prunem,
                        wbits=wbits,
                        nsamples=64,
                        percdamp=0.01,
                        blocksize=128
                    )
                    pruning_applied = True
                else:
                    print("(密集模型)", end=' ')
                    pruning_applied = False
                
                # 进行评估
                ppl = evaluate_with_gemm_hook(
                    model, dataloader, testloader, hook_manager, dataset, 
                    inference_idx=inference_idx, gmp=gmp, sparsity=sparsity
                )
                ppl_results.append(ppl)
                print(f"PPL: {ppl:.4f}")
                
                # 仅在第一次推理时收集 hook 数据（避免内存重复累积）
                if inference_idx == 0:
                    try:
                        collect_hook_data(model, testloader, hook_manager)
                    except Exception as e:
                        print(f"    警告：无法收集 hook 数据: {e}")
                
                # 保存推理数据
                hook_manager.save_inference_data(ppl)
                
                # 只清理模型（dataloader 在外层循环中还需要使用）
                del model
                torch.cuda.empty_cache()
            
            # 输出推理结果统计
            avg_ppl = np.mean(ppl_results)
            print(f"  平均 PPL: {avg_ppl:.4f} (最小: {np.min(ppl_results):.4f}, 最大: {np.max(ppl_results):.4f})")
            
            # 保存评估摘要（必须在删除 hook_manager 前调用）
            hook_manager.save_summary(ppl_results)
            
            # dataset 循环结束后清理 dataloader、testloader 和 hook_manager
            del dataloader, testloader, hook_manager
            torch.cuda.empty_cache()
    
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

    tensors_dirs = find_tensors_dirs("output")
    
    for tensors_dir in tensors_dirs:
        for name in os.listdir(tensors_dir):
            if 'linear' in name:
                if 'head' in name:
                    print(f"重命名: {name} -> {name.replace('72', '12')}")
                    os.rename(os.path.join(tensors_dir, name), os.path.join(tensors_dir, name.replace('72', '12')))
                else:
                    if 'input' in name:
                        newname = 'layer_'+get_after_decoder_layers_simple(name).rsplit('.pt', 1)[0]+'_input.pt'
                        print(f"重命名: {name} -> {newname}")
                        os.rename(os.path.join(tensors_dir, name), os.path.join(tensors_dir, newname))
                    else:
                        newname = 'layer_'+get_after_decoder_layers_simple(name).rsplit('.pt', 1)[0]+'_weight.pt'
                        print(f"重命名: {name} -> {newname}")
                        os.rename(os.path.join(tensors_dir, name), os.path.join(tensors_dir, newname))