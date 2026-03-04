import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

# 设置本地数据路径，可以通过环境变量覆盖
LOCAL_DATA_ROOT = os.environ.get('LOCAL_DATA_ROOT', '/home/zhaojun/proj26/datasets')


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer, data_root=None):
    if data_root is None:
        data_root = LOCAL_DATA_ROOT
    
    # 尝试从本地加载，如果失败则从远程加载
    try:
        traindata = load_dataset(
            'parquet', 
            data_files=os.path.join(data_root, 'wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet'),
            split='train'
        )
        testdata = load_dataset(
            'parquet',
            data_files=os.path.join(data_root, 'wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet'),
            split='train'
        )
    except Exception as e:
        print(f'本地加载失败: {e}，从远程加载...')
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer, data_root=None):
    if data_root is None:
        data_root = LOCAL_DATA_ROOT
    
    # 尝试从本地加载，如果失败则从远程加载
    try:
        import glob
        ptb_files = glob.glob(os.path.join(data_root, 'ptb_text_only/**/*.parquet'), recursive=True)
        if ptb_files:
            # 按照文件名排序来区分 train 和 test
            train_files = [f for f in ptb_files if 'train' in f]
            test_files = [f for f in ptb_files if 'test' in f]
            traindata = load_dataset('parquet', data_files=train_files, split='train') if train_files else load_dataset('parquet', data_files=ptb_files[:1], split='train')
            testdata = load_dataset('parquet', data_files=test_files, split='train') if test_files else load_dataset('parquet', data_files=ptb_files[-1:], split='train')
        else:
            raise FileNotFoundError('未找到本地 PTB 数据')
    except Exception as e:
        print(f'本地加载失败: {e}，从远程加载...')
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer, data_root=None):
    if data_root is None:
        data_root = LOCAL_DATA_ROOT
    
    # 尝试从本地加载，如果失败则从远程加载
    try:
        import glob
        c4_files = glob.glob(os.path.join(data_root, 'c4/**/*.parquet'), recursive=True)
        if c4_files:
            traindata = load_dataset('parquet', data_files=c4_files, split='train')
            valdata = traindata  # 使用同一份数据作为验证集
        else:
            raise FileNotFoundError('未找到本地 C4 数据')
    except Exception as e:
        print(f'本地加载失败: {e}，从远程加载...')
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model='', data_root=None):
    if data_root is None:
        data_root = LOCAL_DATA_ROOT
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer, data_root)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer, data_root)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer, data_root)
