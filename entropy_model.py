# -*- coding: utf-8 -*-
# @Time:   2022/7/4 21:39 
# @Author:  Knight
#本文件基于compressai.entropy_models中的EntropyBottleneck部分源码，延续了里面的命名方式，在某些位置做了一些修改。
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Optional, Tuple

class EntropyBottleneck(nn.Module):
    _offset : Tensor
    def __init__(self,
                 channels: int,
                 *args: Any,
                 tail_mass: float = 1e-9,
                 init_scale: float = 10,
                 filters: Tuple[int, ...] = (3, 3, 3, 3),
                 **kwargs: Any,
                 ):
        #根据给定的参数初始化类及其父类#
        super().__init__(*args, **kwargs)
        self.channels = int(channels)
        self.fileters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        #定义一些参数,channels是通道数，概率全分解的方法利用复合映射的方法拟合出各个通道的概率分布，即每个通道都有一组待优化参数，
        # 对应后续代码中所有待优化参数的第一个维度都是channels
        #scale是待编码元素的一个范围
        #在论文中Appendix6.1中给出概率全分解的超参数设定为K=4， r1=r2=r3=3, r4=1, d1=1,d2=d3=d4=3，对应变换的维度1-> 3-> 3-> 3-> 1
        # filter在self.filter元组的基础上首尾各添加1
        filters = (1, ) + self.filters + (1,)
        scale = self.init_scale ** (1/(len(self.fileters)+1))
        channels = self.channels

        #给待优化参数H，b，a赋初值并将其添加至待优化参数中
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i+1]))
            matrix = torch.Tensor(channels, filters[i+1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i+1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i+1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        #以init为元素构成一个(quantiles.size(0), 1, 1)矩阵
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        #target无需有梯度
        target = np.log(2/self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _get_medians(self) -> Tensor:
        #获取每个通道上的中位值
        medians = self.quantiles[:, 0, 1]
        return medians

    def update(self, force: bool=False):
        if self._offset.numel() > 0 and not force:
            return False
        #medians, minima, maxima格式均为[channels]
        medians = self.quantiles[:, 0, 1]

        minima = medians-self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        # 将minma的最小值设定为0
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1
        #max_length是送入累积概率计算函数中的采样点数
        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)
        #第一项格式为[1, max_length]，第二项格式为[channels, 1, 1]，samples格式为[channel, 1, max_length]
        #相当于对每个通道的待编码数值做一个偏移
        samples = samples[None, :]+pmf_start[:, None, None]

        half = float(0.5)
        #计算c(x+o.5)与c(x-0.5)
        lower = self._logits_cumulative(samples-half, stop_gradient=True)
        upper = self._logits_cumulative(samples+half, stop_gradient=True)
        sign = -torch.sign(lower+upper)
        pmf = torch.abs(torch.sigmoid(sign*upper)-torch.sigmoid(sign*lower))
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, 0])+torch.sigmoid(-upper[:, 0, -1])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        # 记录累计概率的数组的长度应该是记录概率的数组的长度加2
        self._cdf_length = pmf_length+2
        return True
    #熵模型的loss
    def loss(self) -> Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        #输出logits的格式与输入inputs的格式相同，都是[channels, 1, length]stop_gradient指定是否计算待优化参数的梯度
        logits = inputs
        for i in range(len(self.filters)+1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            #矩阵乘法，对应H(k)*x
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f'_bias{i:d}')
            if stop_gradient:
                bias = bias.detach()
            #对应H(k)*x+b(k)
            logits += bias

            #对于所有的k<K,logits = x+a(k)*tanh(H(k)x+b(k))，此处乘法为元素相乘
            if i < len(self.filters):
                factor = getattr(self, f'_factor{i:d}')
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unsused
    #这个装饰器表示在保存模型时跳过这个函数，即此函数不会被转化为torchscript
    def _likelihood(self, inputs: Tensor) ->Tensor:
        half = float(0.5)
        lower = self._logits_cumulative(inputs-half, stop_gradient=False)
        upper = self._logits_cumulative(inputs+half, stop_gradient=False)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        #对应f_K(x) = sigmoid(H(K)x+b(K))，括号里面的已经在_logits_cumulative中计算完成
        likelihood = torch.abs(torch.sigmoid(sign*upper)-torch.sigmoid(sign*lower))
        return likelihood

    def forward(self, x: Tensor, training: Optional[bool]=None) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        #判断是否是torchscript,如果是的话需要做个简单的替代因为有些函数在转化过程中被跳过了
        if not torch.jit.is_scripting():
            # 交换x的前两个维度B和C使第一个维度是C
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            #按理来讲这个inv_perm和perm是一样的，不知道有啥用
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            perm = (1, 2, 3, 0)
            inv_perm = (3, 0, 1, 2)
        #等价于torch.transpose(x, 1, 0).contigious()
        x = x.permute(*perm).contiguous()
        shape = x.size()
        # 待编码的数据,x.zize(0)就是channel
        values = x.reshape(x.sizeo(0), 1, -1)

        #量化操作
        outputs = self.quantize(values, 'noise' if training else 'dequantize', self._get_medians())

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            #添加约束
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            likelihood = torch.zeros_like(outputs)
        #重新转化为输入的数据维度
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        #indexes.shape [1, C, 1, 1]
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()
        #返回的元素格式为[N, C, H, W],每个通道的元素全部为当前通道的索引编号
        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    #展成一个除dim0外全是1维的张量
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)