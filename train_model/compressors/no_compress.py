import torch
import numpy as np
from scipy import stats

from utils.vecs_io import fvecs_read
from utils.vec_np import normalize
from .probabilistic_scalar_compressor import ProbabilisticScalarCompressor

class NoCompressor(object):
    def __init__(self, size, shape, args):
        self.cuda = not args.no_cuda
        self.size = size
        self.shape = shape
        self.users = 1
        # self.k = size // 146 // 2
        self.k1=size//20
        self.k = size // 4

    # def compress(self, vec):
    #     # print('vec.numel()',vec.numel())
    #     vec = vec.view(self.users, -1)
    #     # 为1维
    #     ind = torch.zeros_like(vec)
    #     idx = torch.topk(torch.abs(vec), k=self.k, dim=1)[1]
    #     ind.scatter_(1, idx, 1)
    #     t=vec * ind
    #     # print('(vec * ind).numel()', t.numel())
    #     # print("************vec * ind is*******************", vec * ind,"\n**************")
    #     return t

    def compress(self, vec):
        # print('vec.numel()',vec.numel())
        vec = vec.view(self.users, -1)
        # 为1维
        ind1 = torch.ones_like(vec)
        # # ind = torch.zeros_like(vec)
        # #
        # # idx1 = torch.topk(torch.abs(vec), k=self.k1, dim=1)[1]
        # # ind1.scatter_(1, idx1, 0)
        # t = vec
        #
        # # idx = torch.topk(torch.abs(t), k=self.k, dim=1)[1]
        # # ind.scatter_(1, idx, 1)
        # t = t * ind1


        # print('(vec * ind).numel()', t.numel())
        # print("************vec * ind is*******************", vec * ind,"\n**************")
        return vec*ind1

    def decompress(self, signature):
        # print("##################\n", signature.view(self.shape), "\n##################")
        return signature.view(self.shape)