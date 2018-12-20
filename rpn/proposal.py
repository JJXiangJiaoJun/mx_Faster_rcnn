from __future__ import absolute_import

from mxnet import autograd
from mxnet import gluon, nd
from ..utils.bbox import BBoxCornerToCenter, BBoxClipToImage
from ..utils.coder import NormalizedBoxCenterDecoder

"""
RPN用于生成region proposal类

"""


class RPNProposal(gluon.Block):
    """
    @:parameter
    ------------------
    clip : float
        如果提供，将bbox剪切到这个值
    num_thresh : float
        nms的阈值，用于去除重复的框
   train_pre_nms : int
        训练时对前 train_pre_nms 进行 NMS操作
    train_post_nms : int
        训练时进行NMS后，返回前 train_post_nms 个region proposal
    test_pre_nms : int
        测试时对前 test_pre_nms 进行 NMS操作
    test_post_nms : int
        测试时进行NMS后，返回前 test_post_nms 个region proposal
    min_size : int
        小于 min_size 的 proposal将会被舍弃
    
    stds : tuple of int 
        计算偏移量用的标准差
    
    """

    def __init__(self, clip, nms_thresh, train_pre_nms, train_post_nms,
                 test_pre_nms, test_post_nms, min_size, stds, **kwargs):
        super(RPNProposal, self).__init__(**kwargs)
        self._clip = clip
        self._nms_thresh = nms_thresh
        self._train_pre_nms = train_pre_nms
        self._train_post_nms = train_post_nms
        self._test_pre_nms = test_pre_nms
        self._test_post_nms = test_post_nms
        self._min_size = min_size
        self._bbox_decoder = NormalizedBoxCenterDecoder(stds=stds, clip=clip)
        self._cliper = BBoxClipToImage()
        self._bbox_tocenter = BBoxCornerToCenter(axis=-1, split=False)

    """
    @:parameter
    scores : （B,N,1) 
        通过RPN预测的得分输出(sigmoid之后) (0,1)
    offsets : ndarray (B,N,4)
        通过RPN预测的锚框偏移量
    anchors : ndarray (B,N,4)
        生成的默认锚框，坐标编码方式为 Corner
    img : ndarray (B,C,H,W)
        图像的张量，用来剪切锚框
    
    @:returns
    
    
    """

    def forward(self, scores, offsets, anchors, img):
        # 训练和预测的处理流程不同
        if autograd.is_training():
            pre_nms = self._train_pre_nms
            post_nms = self._train_post_nms
        else:
            pre_nms = self._test_pre_nms
            post_nms = self._test_post_nms
        with autograd.pause():
            # 将预测的偏移量加到anchors中
            rois = self._bbox_decoder(offsets, self._bbox_tocenter(anchors))
            rois = self._cliper(rois, img)

            # 下面将所有尺寸小于设定最小值的ROI去除
            x_min, y_min, x_max, y_max = nd.split(rois, num_outputs=4, axis=-1)
            width = x_max - x_min
            height = y_max - y_min
            invalid_mask = (width < self._min_size) + (height < self._min_size)

            # 将对应位置的score 设为-1
            scores = nd.where(invalid_mask, nd.ones_like(scores) * -1, scores)
            invalid_mask = nd.repeat(invalid_mask, repeats=4, axis=-1)
            rois = nd.where(invalid_mask, nd.ones_like(rois) * -1, rois)

            # 下面进行NMS操作
            pre = nd.concat(scores, rois, dim=-1)
            pre = nd.contrib.box_nms(pre, overlap_thresh=self._nms_thresh, topk=pre_nms,
                                     coord_start=1, score_index=0, id_index=-1, force_suppress=True)
            # 下面进行采样
            result = nd.slice_axis(pre,axis=1, begin=0, end=post_nms)
            rpn_score = nd.slice_axis(result, axis=-1, begin=0, end=1)
            rpn_bbox = nd.slice_axis(result, axis=-1, begin=1, end=None)

        return rpn_score, rpn_bbox
