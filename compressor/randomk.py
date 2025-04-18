# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 25/03/2024 20:21
# @Author       : C. Zhang
# @File         : randomk.py
# @Affiliation  : Shandong University
import torch
import sys
sys.path.append('../../')
from fedgcc.compressor import Compressor


def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))
    indices = torch.randperm(numel, device=tensor.device)[:k]
    values = tensor[indices]
    return indices, values


class RandomKCompressor(Compressor):
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

    def __init__(self, compress_ratio):
        super().__init__()
        self.global_step = 0
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name=None):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        h = sum(bytes(name, encoding='utf8'), self.global_step)
        self.global_step += 1
        torch.manual_seed(h)
        indices, values = sparsify(tensor, self.compress_ratio)

        ctx = indices, tensor.numel(), tensor.size()
        return [values, indices], ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, numel, shape = ctx
        values, _ = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)