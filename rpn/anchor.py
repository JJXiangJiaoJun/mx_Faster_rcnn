"""RPN网络 anchors"""

import numpy as np
from mxnet import gluon
from mxnet import nd
import mxnet as mx

class RPNAnchorGenerator(gluon.Block):
    """
    @输入参数
    stride:int              
        特征图的每个像素感受野大小，通常为原图和特征图尺寸比例
    base_size:int           
        默认大小
    ratios:int              
        宽高比
    scales:int              
        大小比例
        
        每个锚框为   width = base_size*size/sqrt(ratio)  
                    height = base_size*size*sqrt(ratio)
        
    alloc_size:(int,int)          
        默认的特征图大小(H,W)，以后每次生成直接索引切片
    """

    def __init__(self, stride, base_size, ratios, scales, alloc_size, **kwargs):
        super(RPNAnchorGenerator, self).__init__(**kwargs)
        if not base_size:
            raise ValueError("Invalid base_size: {}.".format(base_size))
        if not isinstance(ratios, (tuple, list)):
            ratios = [ratios]
        if not isinstance(scales, (tuple, list)):
            scales = [scales]

        anchors = self._generate_anchors(stride, base_size, ratios, scales, alloc_size)
        self._num_depth = len(ratios) * len(scales)
        self.anchors = self.params.get_constant('anchor_', anchors)

    @property
    def num_depth(self):
        return self._num_depth

    def _generate_anchors(self, stride, base_size, ratios, scales, alloc_size):
        # 计算中心点坐标
        px, py = (base_size - 1) * 0.5, (base_size - 1) * 0.5
        base_sizes = []
        for r in ratios:
            for s in scales:
                size = base_size * base_size / r
                ws = np.round(np.sqrt(size))
                w = (ws * s - 1) * 0.5
                h = (np.round(ws * r) * s - 1) * 0.5
                base_sizes.append([px - w, py - h, px + w, py + h])
        # 每个像素的锚框
        base_sizes = np.array(base_sizes)

        # 下面进行偏移量的生成
        width, height = alloc_size
        offset_x = np.arange(0, width * stride, stride)
        offset_y = np.arange(0, height * stride, stride)
        offset_x, offset_y = np.meshgrid(offset_x, offset_x)
        # 生成(H*W,4)
        offset = np.stack((offset_x.ravel(), offset_y.ravel(),
                           offset_x.ravel(), offset_y.ravel()), axis=1)

        # 下面广播到每一个anchor中    (1,N,4) + (M,1,4)
        anchors = base_sizes.reshape((1, -1, 4)) + offset.reshape((-1, 1, 4))
        anchors = anchors.reshape((1, 1, width, height, -1)).astype(np.float32)
        return anchors

    # 对原始生成的锚框进行切片操作
    # pylint: disable=arguments-differ
    def forward(self,x):
        """Slice anchors given the input image shape.

        Inputs:
            - **x**: input tensor with (1 x C x H x W) shape.
        Outputs:
            - **out**: output anchor with (1, N, 4) shape. N is the number of anchors.

        """
        anchors = self.anchors.value
        anchors = anchors.as_in_context(x.context)
        a = nd.slice_like(anchors, x * 0, axes=(2, 3))
        return a.reshape((1, -1, 4))
