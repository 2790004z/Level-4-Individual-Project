# The based unit of graph convolutional networks.
 
import torch
import torch.nn as nn
 
# GCN模块 主要是一个Conv2d和一个einsum
class ConvTemporalGraphical(nn.Module):
 
    r"""The basic module for applying a graph convolution.
    主要是空间域卷积的结构和前向传播方法
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
 
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()
        # 这个卷积核指的是空间上的kernel_size,为3，也等于分区策略划分的子集数K
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
 
 
    # forward()函数完成图卷积操作,x由（64,3,300,18）变成（64,64,300,18）
    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
 
        x = self.conv(x)
        # (64,192,300,18)
        n, kc, t, v = x.size()
        # (64,3,64,300,18)
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        # (64,64,300,18)
        # 此处的k消失的原因：在k维度上进行了求和操作,也即是x在邻接矩阵A的3个不同的子集上进行乘机操作再进行求和,
        # 对应于论文中的公式10
        x = torch.einsum('nkctv,kvw->nctw', (x, A))  # 爱因斯坦约定求和法
        # contiguous()把tensor x变成在内存中连续分布的形式
        return x.contiguous(), A