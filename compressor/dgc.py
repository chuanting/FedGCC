# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 25/03/2024 20:21
# @Author       : C. Zhang
# @File         : dgc.py
# @Affiliation  : Shandong University
import torch
import sys
sys.path.append('../../')
from fedgcc.compressor import Compressor


class DGCCompressor(Compressor):

    def __init__(self, compress_ratio):
        super().__init__(tensors_size_are_same=False)
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name=None):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        sample_shape = [max(1, int(numel * 0.01))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        k = max(1, int(numel * self.compress_ratio * 0.01))
        vals, indices = torch.topk(sample_tensor.abs(), k)

        thr = vals.min()
        mask = tensor.abs() >= thr
        selected = mask.sum()

        for _ in range(10):
            if selected > 1.3 * numel * self.compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * self.compress_ratio:
                thr = 0.7 * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        indices, = torch.where(mask)
        values = tensor[indices]

        tensor_compressed = values, indices
        ctx = shape, mask, numel
        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        values, indices = tensor_compressed
        shape, _, numel = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)