"""
GEMM/SpGEMM Hook utilities for capturing tensor operations
"""
import torch
import os
import json
from collections import defaultdict
from pathlib import Path


class GEMMRecorder:
    """Records GEMM and SpGEMM operations during model execution"""
    
    def __init__(self, output_dir, model_name):
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.records = []
        self.call_counter = defaultdict(int)
        self.enabled = True
        
    def record_operation(self, op_type, input1, input2, output=None, **kwargs):
        """Record a GEMM/SpGEMM operation"""
        if not self.enabled:
            return
            
        op_id = self.call_counter[op_type]
        self.call_counter[op_type] += 1
        
        record = {
            'op_type': op_type,
            'call_id': op_id,
            'input1_shape': list(input1.shape) if hasattr(input1, 'shape') else None,
            'input2_shape': list(input2.shape) if hasattr(input2, 'shape') else None,
            'output_shape': list(output.shape) if output is not None and hasattr(output, 'shape') else None,
            'input1_dtype': str(input1.dtype) if hasattr(input1, 'dtype') else None,
            'input2_dtype': str(input2.dtype) if hasattr(input2, 'dtype') else None,
            'input1_device': str(input1.device) if hasattr(input1, 'device') else None,
            'input2_device': str(input2.device) if hasattr(input2, 'device') else None,
            'input1_sparse': input1.is_sparse if hasattr(input1, 'is_sparse') else False,
            'input2_sparse': input2.is_sparse if hasattr(input2, 'is_sparse') else False,
        }
        
        # Add any additional metadata
        record.update(kwargs)
        
        self.records.append(record)
        
        # Save tensors
        self._save_tensors(op_type, op_id, input1, input2, output)
        
    def _save_tensors(self, op_type, op_id, input1, input2, output):
        """Save input and output tensors"""
        prefix = f"{op_type}_call_{op_id:04d}"
        
        try:
            # Save input tensors
            if input1 is not None:
                input1_path = self.output_dir / f"{prefix}_input1.pt"
                if input1.is_sparse:
                    # Save sparse tensor as COO format
                    torch.save({
                        'indices': input1._indices().cpu(),
                        'values': input1._values().cpu(),
                        'shape': input1.shape
                    }, input1_path)
                else:
                    torch.save(input1.detach().cpu(), input1_path)
            
            if input2 is not None:
                input2_path = self.output_dir / f"{prefix}_input2.pt"
                if input2.is_sparse:
                    torch.save({
                        'indices': input2._indices().cpu(),
                        'values': input2._values().cpu(),
                        'shape': input2.shape
                    }, input2_path)
                else:
                    torch.save(input2.detach().cpu(), input2_path)
            
            # Save output tensor
            if output is not None:
                output_path = self.output_dir / f"{prefix}_output.pt"
                if output.is_sparse:
                    torch.save({
                        'indices': output._indices().cpu(),
                        'values': output._values().cpu(),
                        'shape': output.shape
                    }, output_path)
                else:
                    torch.save(output.detach().cpu(), output_path)
        except Exception as e:
            print(f"Warning: Failed to save tensors for {prefix}: {e}")
    
    def save_metadata(self):
        """Save metadata JSON file"""
        metadata_path = self.output_dir / "gemm_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'records': self.records,
                'summary': {
                    'total_operations': len(self.records),
                    'operations_by_type': dict(self.call_counter)
                }
            }, f, indent=2)
        print(f"Saved GEMM metadata to {metadata_path}")
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False


# Global recorder instance
_recorder = None


def get_recorder():
    """Get the global recorder instance"""
    return _recorder


def init_recorder(output_dir, model_name):
    """Initialize the global recorder"""
    global _recorder
    _recorder = GEMMRecorder(output_dir, model_name)
    return _recorder


def finalize_recorder():
    """Finalize and save recorder data"""
    global _recorder
    if _recorder is not None:
        _recorder.save_metadata()


# Monkey-patch torch functions to record operations
_original_mm = torch.mm
_original_matmul = torch.matmul
_original_einsum = torch.einsum
_original_spmm = getattr(torch, 'spmm', None) or getattr(torch.sparse, 'mm', None)
_original_sparse_mm = getattr(torch.sparse, 'mm', None)
_original_addmm = torch.addmm
_original_bmm = torch.bmm


def _mm_wrapper(input, mat2, *, out=None):
    """Wrapper for torch.mm"""
    result = _original_mm(input, mat2, out=out)
    if _recorder is not None and _recorder.enabled:
        _recorder.record_operation('torch.mm', input, mat2, result)
    return result


def _matmul_wrapper(input, other, *, out=None):
    """Wrapper for torch.matmul"""
    result = _original_matmul(input, other, out=out)
    if _recorder is not None and _recorder.enabled:
        _recorder.record_operation('torch.matmul', input, other, result)
    return result


def _einsum_wrapper(equation, *operands):
    """Wrapper for torch.einsum"""
    result = _original_einsum(equation, *operands)
    if _recorder is not None and _recorder.enabled and len(operands) >= 2:
        _recorder.record_operation('torch.einsum', operands[0], operands[1], result, equation=equation)
    return result


def _spmm_wrapper(mat1, mat2):
    """Wrapper for torch.spmm (deprecated, use torch.sparse.mm)"""
    # Call original function to avoid recursion
    result = _original_spmm(mat1, mat2) if _original_spmm else mat1.mm(mat2)
    if _recorder is not None and _recorder.enabled:
        _recorder.record_operation('torch.spmm', mat1, mat2, result)
    return result


def _sparse_mm_wrapper(mat1, mat2):
    """Wrapper for torch.sparse.mm"""
    result = _original_sparse_mm(mat1, mat2) if _original_sparse_mm else mat1.mm(mat2)
    if _recorder is not None and _recorder.enabled:
        _recorder.record_operation('torch.sparse.mm', mat1, mat2, result)
    return result


def _addmm_wrapper(bias, input, mat2, *, beta=1, alpha=1, out=None):
    """Wrapper for torch.addmm"""
    result = _original_addmm(bias, input, mat2, beta=beta, alpha=alpha, out=out)
    if _recorder is not None and _recorder.enabled:
        _recorder.record_operation('torch.addmm', input, mat2, result, bias_shape=list(bias.shape))
    return result


def _bmm_wrapper(input, mat2, *, out=None):
    """Wrapper for torch.bmm"""
    result = _original_bmm(input, mat2, out=out)
    if _recorder is not None and _recorder.enabled:
        _recorder.record_operation('torch.bmm', input, mat2, result)
    return result


def install_hooks():
    """Install GEMM hooks by monkey-patching torch functions"""
    torch.mm = _mm_wrapper
    torch.matmul = _matmul_wrapper
    torch.einsum = _einsum_wrapper
    if hasattr(torch, 'spmm'):
        torch.spmm = _spmm_wrapper
    if hasattr(torch.sparse, 'mm'):
        torch.sparse.mm = _sparse_mm_wrapper
    torch.addmm = _addmm_wrapper
    torch.bmm = _bmm_wrapper
    print("GEMM hooks installed")


def uninstall_hooks():
    """Restore original torch functions"""
    torch.mm = _original_mm
    torch.matmul = _original_matmul
    torch.einsum = _original_einsum
    if hasattr(torch, 'spmm') and _original_spmm:
        if hasattr(torch, 'spmm'):
            torch.spmm = _original_spmm
    if _original_sparse_mm:
        if hasattr(torch.sparse, 'mm'):
            torch.sparse.mm = _original_sparse_mm
    torch.addmm = _original_addmm
    torch.bmm = _original_bmm
    print("GEMM hooks uninstalled")
