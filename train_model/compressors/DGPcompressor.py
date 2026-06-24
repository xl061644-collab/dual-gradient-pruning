import torch
import numpy as np
from scipy import stats

from utils.vecs_io import fvecs_read
from utils.vec_np import normalize
from .probabilistic_scalar_compressor import ProbabilisticScalarCompressor

class DGPCompressor(object):
    def __init__(self, size, shape, args):
        self.cuda = not args.no_cuda
        self.size = size
        self.shape = shape
        self.users = 1
        self.k = (size*2)// 5
        self.k1= (size)// 20

    def compress(self,vec):
        vec = vec.view(self.users, -1)
        ind1 = torch.ones_like(vec)
        idx = torch.topk(torch.abs(vec), k=(self.k), dim=1)[1]
        idx1 = torch.topk(torch.abs(vec), k=self.k1, dim=1)[1]
        ind1.scatter_(1, idx1, 0)
        vec = vec * ind1
        ind = torch.zeros_like(vec)
        idx = torch.topk(torch.abs(vec), k=self.k // 2, dim=1)[1]
        ind.scatter_(1, idx, 1)
        return vec * ind

    def decompress(self, signature):
        # print("##################\n", signature.view(self.shape), "\n##################")
        return signature.view(self.shape)

