"""Region Proposal 标注工具."""
from __future__ import absolute_import

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd
from ..utils.bbox import BBoxSplit
from ..utils.coder import SigmoidClassEncoder, NormalizedBoxCenterEncoder


####只支持 batch_size=1

class RPNTargetSampler(gluon.Block):
    """
    @输入参数
    -----------------
    
    num_sample  : int
        RPN采样的训练样本总数
    pos_iou_thresh  :   float
        IOU 大于 pos_iou_thresh 的锚框将被视为正类
    neg_iou_thresh  :   float
        IOU 小于 neg_iou_thresh 的锚框将被视为负类
    pos_ratio   :   float
        采样输出中正样本比例，最终的正样本数量为 num_sample*pos_ratio
    """

    def __init__(self, num_sample, pos_iou_thresh, neg_iou_thresh, pos_ratio, **kwargs):
        super(RPNTargetSampler, self).__init__(**kwargs)
        self._pos_iou_thresh = pos_iou_thresh
        self._num_sample = num_sample
        self._neg_iou_thresh = neg_iou_thresh
        self._max_pos = int(np.round(pos_ratio * num_sample))
        self._eps = np.spacing(np.float32(1.0))

    """
    @输入参数
    ----------------------
    
    ious : ndarray  
        (N,M) 通过box_iou 生成的交并比

    @:return
    ------------------
    
    samples :  ndarray
        (N,)  采样的锚框                     1： pos  0:ignore    -1:neg
    matches :   ndarray
        (N,)   匹配的ground truth 索引       [0,M）

    """

    def forward(self, ious):

        matches = nd.argmax(ious, axis=-1)
        # 每个锚框最高得分
        max_iou_pre_anchor = nd.max(ious, axis=-1)
        # 将所有锚框都初始化为0，ignore
        samples = nd.zeros_like(max_iou_pre_anchor)

        # 计算每个ground_truth 的最高iou
        max_all_ious = nd.max(ious, axis=0, keepdims=True)
        # 标记处mask中最高分值的那一行为1
        mask = nd.broadcast_greater(ious + self._eps, max_all_ious)
        mask = nd.sum(mask, axis=-1)
        # 将最高分数的锚框标记为 1 正类
        samples = nd.where(mask, nd.ones_like(samples), samples)

        # 下面标记大于 pos_iou_thresh的样本为正例
        samples = nd.where(max_iou_pre_anchor > self._pos_iou_thresh, nd.ones_like(samples), samples)

        # 标记小于neg_iou_thresh的样本为负类
        tmp = (max_iou_pre_anchor < self._neg_iou_thresh) * (max_iou_pre_anchor > 0)

        samples = nd.where(tmp, nd.ones_like(samples) * -1, samples)
        # 将其转换为 numnpy
        samples = samples.asnumpy()
        # 下面进行采样
        # 首先对正样本进行采样
        num_pos = int((samples > 0).sum())
        if num_pos > self._max_pos:
            discard_indices = np.random.choice(
                np.where((samples > 0))[0], size=(num_pos - self._max_pos), replace=False
            )
            samples[discard_indices] = 0  # 将多余部分设置为忽略
        num_neg = int((samples < 0).sum())
        max_neg = self._num_sample - min(self._max_pos, num_pos)

        if num_neg > max_neg:
            discard_indices = np.random.choice(
                np.where((samples < 0))[0], size=(num_neg - max_neg), replace=False
            )
            samples[discard_indices] = 0

        # 最后将其转化为ndarray
        samples = nd.array(samples, ctx=matches.context)
        return samples, matches


class RPNTargetGenerator(gluon.Block):
    """
    @parameter
    ------------------
    num_sample  : int
        RPN采样的训练样本总数
    pos_iou_thresh  :   float
        IOU 大于 pos_iou_thresh 的锚框将被视为正类
    neg_iou_thresh  :   float
        IOU 小于 neg_iou_thresh 的锚框将被视为负类
    pos_ratio   :   float
        采样输出中正样本比例，最终的正样本数量为 num_sample*pos_ratio
    
    stds : tuple of float
        标注bbox偏移量时使用的标准差
    means : tuple of float
        标注bbox偏移量时使用的均值
    
    """

    def __init__(self, num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.25, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.), **kwargs):
        super(RPNTargetGenerator, self).__init__(**kwargs)
        self._sampler = RPNTargetSampler(num_sample, pos_iou_thresh, neg_iou_thresh, pos_ratio)
        self._spliter = BBoxSplit(axis=-1)
        self._cls_encoder = SigmoidClassEncoder()
        self._bbox_encoder = NormalizedBoxCenterEncoder(stds, means)

    """
    @:parameter
    
    --------------------
    bboxes    : (M,4)
        ground_truth 坐标
    
    anchors  : (N,4)
        生成的anchor 坐标
    
    height   :   int
        图片高度（用来去除超出边界的锚框）
    width    :   int
        图片宽度（用来去除超出边界的锚框）
    
    @:returns
    ----------------
    cls_label   : (N,)  类别标签  1:pos 0:neg -1:ignore
    bbox_label  : (N,4) 偏移量Normalized
    bbox_mask   : (N,4) 只有正类的mask>0
    
    """

    def forward(self, bboxes, anchors, height, width):
        # 标注ious
        with autograd.pause():
            ious = mx.nd.contrib.box_iou(anchors, bboxes)
            # 去除无效的锚框(超出边界的)
            x_min, y_min, x_max, y_max = self._spliter(anchors)
            invalid_mask = (x_min < 0) + (y_min < 0) + (x_max >= width) + (y_max >= height)
            # 将所有无效锚框的ious设为-1
            invalid_mask = nd.repeat(invalid_mask, repeats=bboxes.shape[0], axis=-1)
            ious = nd.where(invalid_mask > 0, nd.ones_like(ious) * -1, ious)

            # 对锚框进行采样
            samples, matches = self._sampler(ious)
            # 下面进行标注

            cls_label, _ = self._cls_encoder(samples)
            targets, masks = self._bbox_encoder(samples.expand_dims(axis=0), matches.expand_dims(axis=0),
                                                anchors.expand_dims(axis=0), bboxes.expand_dims(axis=0))

        return cls_label, targets[0], masks[0]
