import os
import sys
import json
import pickle
import torch
import argparse
from pathlib import Path

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
sys.path.append(root_path + 'src')
os.chdir(root_path)

from hin_loader import HIN
from evaluation import eval_logits, eval_and_save
from config import HGSLConfig
from HGSL import HGSL
import util_funcs as uf


class SPMMHook:
    """Hook for capturing torch.spmm inputs and outputs"""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spmm_calls = []
        self.call_count = 0
        # Store original spmm function
        self.original_spmm = torch.spmm
        
    def hook(self, sparse_mat, dense_mat):
        """Replacement function for torch.spmm"""
        # Save both inputs
        is_sparse = sparse_mat.is_sparse
        call_data = {
            'call_id': self.call_count,
            'sparse_shape': tuple(sparse_mat.shape),
            'sparse_is_sparse': is_sparse,
            'dense_shape': tuple(dense_mat.shape),
        }
        
        if is_sparse:
            try:
                call_data['sparse_nnz'] = sparse_mat._nnz().item()
            except:
                call_data['sparse_nnz'] = 'unknown'
        
        # Save sparse matrix
        sparse_path = self.output_dir / f'spmm_call_{self.call_count:04d}_sparse.pt'
        sparse_cpu = sparse_mat.cpu() if sparse_mat.is_cuda else sparse_mat
        torch.save(sparse_cpu, sparse_path)
        call_data['sparse_path'] = str(sparse_path)
        
        # Save dense matrix
        dense_path = self.output_dir / f'spmm_call_{self.call_count:04d}_dense.pt'
        dense_cpu = dense_mat.cpu() if dense_mat.is_cuda else dense_mat
        torch.save(dense_cpu, dense_path)
        call_data['dense_path'] = str(dense_path)
        
        # Call original spmm
        output = self.original_spmm(sparse_mat, dense_mat)
        output_path = self.output_dir / f'spmm_call_{self.call_count:04d}_output.pt'
        output_cpu = output.cpu() if output.is_cuda else output
        torch.save(output_cpu, output_path)
        call_data['output_path'] = str(output_path)
        call_data['output_shape'] = tuple(output.shape)
        
        self.spmm_calls.append(call_data)
        self.call_count += 1
        
        return output
    
    def save_metadata(self):
        """Save metadata about all SPMM calls"""
        metadata_path = self.output_dir / 'spmm_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.spmm_calls, f, indent=2)
        print(f"✓ Saved SPMM metadata: {metadata_path}")
        
        summary_path = self.output_dir / 'spmm_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Total SPMM calls: {len(self.spmm_calls)}\n\n")
            for i, call in enumerate(self.spmm_calls):
                f.write(f"Call {i}:\n")
                f.write(f"  Sparse shape: {call['sparse_shape']}\n")
                f.write(f"  Dense shape: {call['dense_shape']}\n")
                f.write(f"  Output shape: {call['output_shape']}\n\n")
        print(f"✓ Saved SPMM summary: {summary_path}")


def evaluate_model(dataset, gpu_id=0, model_path=None, output_dir=None):
    """
    Evaluate trained HGSL model with SPMM hook
    
    Args:
        dataset: Dataset name ('acm', 'dblp', or 'yelp')
        gpu_id: GPU id to use
        model_path: Path to saved model checkpoint
        output_dir: Directory to save SPMM inputs/outputs
    """
    uf.seed_init(0)
    uf.shell_init(gpu_id=gpu_id)
    
    cf = HGSLConfig(dataset)
    cf.dev = torch.device("cuda:0" if gpu_id >= 0 else "cpu")
    
    # Create output directory
    if output_dir is None:
        output_dir = root_path + 'output'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📊 Loading dataset: {dataset}")
    g = HIN(cf.dataset).load_mp_embedding(cf)
    features, adj, mp_emb, train_x, train_y, val_x, val_y, test_x, test_y = g.to_torch(cf)
    
    # Create model
    print(f"🏗️  Creating model...")
    model = HGSL(cf, g)
    model.to(cf.dev)
    
    # Load checkpoint
    if model_path is None:
        # Auto-find latest model for this dataset
        import glob
        pattern = root_path + f'temp/HGSL/{dataset}/*.pt'
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found for dataset {dataset} in {pattern}")
        model_path = checkpoints[0]
    
    print(f"📦 Loading checkpoint: {model_path}")
    state_dict = torch.load(model_path, map_location=cf.dev)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Setup SPMM hook
    print(f"🎣 Setting up SPMM hook...")
    spmm_hook = SPMMHook(output_dir)
    
    # Replace torch.spmm with hooked version
    original_spmm = torch.spmm
    torch.spmm = spmm_hook.hook
    
    # Forward pass
    print(f"🔄 Running forward pass...")
    with torch.no_grad():
        try:
            logits, adj_new = model(features, adj, mp_emb)
        finally:
            # Restore original torch.spmm
            torch.spmm = original_spmm
    
    # Evaluate
    print(f"\n📈 Evaluating results...")
    test_f1, test_mif1 = eval_logits(logits, test_x, test_y)
    val_f1, val_mif1 = eval_logits(logits, val_x, val_y)
    
    # Save results
    results = {
        'dataset': dataset,
        'model_path': model_path,
        'test_f1': float(test_f1),
        'test_micro_f1': float(test_mif1),
        'val_f1': float(val_f1),
        'val_micro_f1': float(val_mif1),
        'num_spmm_calls': len(spmm_hook.spmm_calls),
    }
    
    results_file = Path(output_dir) / f'{dataset}_eval_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation Results:")
    print(f"   Test F1: {test_f1:.4f}")
    print(f"   Test Micro-F1: {test_mif1:.4f}")
    print(f"   Val F1: {val_f1:.4f}")
    print(f"   Val Micro-F1: {val_mif1:.4f}")
    print(f"   Total SPMM calls: {len(spmm_hook.spmm_calls)}")
    
    # Save SPMM data
    spmm_hook.save_metadata()
    
    # Save logits and adjacency
    torch.save(logits.cpu(), Path(output_dir) / f'{dataset}_logits.pt')
    torch.save(adj_new.cpu(), Path(output_dir) / f'{dataset}_adj_new.pt')
    print(f"✓ Saved logits and adjacency matrix")
    
    # Save results
    print(f"✓ Saved evaluation results: {results_file}")
    print(f"✓ All outputs saved to: {output_dir}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HGSL model with SPMM hook")
    parser.add_argument('--dataset', type=str, default='dblp', 
                        help='Dataset name (acm, dblp, or yelp)')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='GPU id to use')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (auto-finds latest if not provided)')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/nizhj/gnn_sparsity/HGSL/output',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    
    results = evaluate_model(
        dataset=args.dataset,
        gpu_id=args.gpu_id,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
