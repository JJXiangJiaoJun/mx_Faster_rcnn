from __future__ import absolute_import

import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd
import numpy as np
from ..utils.coder import NormalizedPerClassBoxCenterEncoder, MultiClassEncoder


class RCNNTargetSampler(gluon.Block):
    """
    @:parameter
    ------------
    num_images : int
        每个batch的图片数，目前仅支持1
    num_inputs : int
        输入的RoI 数量
    num_samples : int
        输出的采样 RoI 数量
    pos_thresh : float
        正类样本阈值
        
    pos_ratio : float
        采样正样本的比例
        
    max_gt_box : int
    
    
    """

    def __init__(self, num_images, num_inputs, num_samples, pos_thresh, pos_ratio, max_gt_box, **kwargs):
        super(RCNNTargetSampler, self).__init__(**kwargs)
        self._num_images = num_images
        self._num_inputs = num_inputs
        self._num_samples = num_samples
        self._pos_thresh = pos_thresh
        self._pos_ratios = pos_ratio
        self._max_pos = int(np.round(num_samples * pos_ratio))
        self._max_gt_box = max_gt_box

    def forward(self, rois, scores, gt_bboxes):
        """
        @:parameter
        -----------
        rois : ndarray (B,self._num_inputs,4)
            RPN输出的roi区域坐标，Corner

        scores : ndarray (B,self._num_inputs,1)
            RPN输出的roi区域分数，(0,1) -1表示忽略

        gt_bboxes:ndarray (B,M,4)
            ground truth box 坐标

        @:returns
        -----------
        new_rois : ndarray (B,self._num_samples,4)
            采样后的RoI区域
        new_samples : ndarray (B,self._num_samples,1)
            采样后RoI区域的标签 1:pos -1:neg 0:ignore
        new_matches : ndarray (B,self._num_samples,1)
            采样后的RoI匹配的锚框编号 [0,M)
        
        """

        new_rois, new_samples, new_matches = [], [], []

        # 对每个batch分别进行处理
        for i in range(self._num_images):
            roi = nd.squeeze(nd.slice_axis(rois, axis=0, begin=i, end=i + 1), axis=0)
            score = nd.squeeze(nd.slice_axis(scores, axis=0, begin=i, end=i + 1), axis=0)
            gt_bbox = nd.squeeze(nd.slice_axis(gt_bboxes, axis=0, begin=i, end=i + 1), axis=0)

            # 将ground truth的分数设置为1 形状为(M,1)
            gt_score = nd.ones_like(nd.sum(gt_bbox, axis=-1, keepdims=True))

            # 将ground truth 和 roi 拼接 (N+M,4) (N+m,1)
            roi = nd.concat(roi, gt_bbox, dim=0)
            score = nd.concat(score, gt_score, dim=0).squeeze(axis=-1)

            # 计算iou   (N+M,M)
            iou = nd.contrib.box_iou(roi, gt_bbox, format='corner')
            # (N+M,)
            iou_max = nd.max(iou, axis=-1)
            # (N+M,)  与哪个ground truth 匹配
            iou_argmax = nd.argmax(iou, axis=-1)

            # 将所有的标记为 2 neg
            mask = nd.ones_like(iou_argmax) * 2
            # 标记ignore 为 0
            mask = nd.where(score < 0, nd.zeros_like(mask), mask)

            # 将正类标记为 3 pos
            pos_idx = (iou_max >= self._pos_thresh)

            mask = nd.where(pos_idx, nd.ones_like(mask) * 3, mask)

            # 下面进行shuffle操作
            rand = nd.random.uniform(0, 1, shape=(self._num_inputs + self._max_gt_box,))
            # 取前面 N+M 个 对mask 做shuffle操作
            rand = nd.slice_like(rand, mask)
            # shuffle 操作后的 index
            index = nd.argsort(rand)
            # 将三个结果进行shuffle
            mask = nd.take(mask, index)
            iou_argmax = nd.take(iou_argmax, index)

            # 下面进行采样
            # 排序 3:pos 2:neg 0:ignore
            order = nd.argsort(mask, is_ascend=False)
            # 取topk个作为正例
            topk = nd.slice_axis(order, axis=0, begin=0, end=self._max_pos)
            # 下面取出相对应的值
            pos_indices = nd.take(index, topk)
            pos_samples = nd.take(mask, topk)
            pos_matches = nd.take(iou_argmax, topk)

            # 下面将原来的标签改了
            pos_samples = nd.where(pos_samples == 3, nd.ones_like(pos_samples), pos_samples)
            pos_samples = nd.where(pos_samples == 2, nd.ones_like(pos_samples) * -1, pos_samples)

            index = nd.slice_axis(index, axis=0, begin=self._max_pos, end=None)
            mask = nd.slice_axis(mask, axis=0, begin=self._max_pos, end=None)
            iou_argmax = nd.slice_axis(iou_argmax, axis=0, begin=self._max_pos, end=None)

            # 对负样本进行采样
            # neg 2---->4
            mask = nd.where(mask == 2, nd.ones_like(mask) * 4, mask)
            order = nd.argsort(mask, is_ascend=False)
            num_neg = self._num_samples - self._max_pos
            bottomk = nd.slice_axis(order, axis=0, begin=0, end=num_neg)

            neg_indices = nd.take(index, bottomk)
            neg_samples = nd.take(mask, bottomk)
            neg_matches = nd.take(iou_argmax, topk)

            neg_samples = nd.where(neg_samples == 3, nd.ones_like(neg_samples), neg_samples)
            neg_samples = nd.where(neg_samples == 4, nd.ones_like(neg_samples) * -1, neg_samples)

            # 输出
            new_idx = nd.concat(pos_indices, neg_indices, dim=0)
            new_sample = nd.concat(pos_samples, neg_samples, dim=0)
            new_match = nd.concat(pos_matches, neg_matches, dim=0)

            new_rois.append(roi.take(new_idx))
            new_samples.append(new_sample)
            new_matches.append(new_match)

        new_rois = nd.stack(*new_rois, axis=0)
        new_samples = nd.stack(*new_samples, axis=0)
        new_matches = nd.stack(*new_matches, axis=0)

        return new_rois, new_samples, new_matches


class RCNNTargetGenerator(gluon.Block):
    """
    @:parameter
    -----------
    num_class : int
        总类别数，不包括背景
    means : tuple of float
    
    stds :  tuple of float
    
    
    
    """

    def __init__(self, num_class, means=(0., 0., 0., 0.), stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(RCNNTargetGenerator, self).__init__(**kwargs)
        self._num_class = num_class
        self._bbox_encoder = NormalizedPerClassBoxCenterEncoder(num_class, stds, means)
        self._cls_encoder = MultiClassEncoder()

    def forward(self, rois, samples, matches, gt_boxes, gt_labels):
        """
        :param rois: (B,N,4) 
            采样之后的roi坐标
        :param samples: (B,N,1)
            采样之后的标签 +1:pos / -1:neg  0:ignore
        :param matches: 
            匹配的ground truth索引 [0，M)
        :param gt_boxes: (B,M,4)
        :param gt_labels: (B,M,)
        
        
        :return: 
        cls_target : (B,N)
            [0，num_class+1)类别标签，包括背景
        box_target:  (B,N,C,4)
        
        bbox_mask : (B,N,C,4)
        
        """
        with autograd.pause():
            # 首先进行采样

            # 对类别编码
            cls_target = self._cls_encoder(samples, matches, gt_labels)
            # 对bbox进行偏移量标注
            bbox_target, bbox_mask = self._bbox_encoder(samples, matches, rois, gt_labels, gt_boxes)

            # modify shapes to match predictions
            # box (C, B, N, 4) -> (B, N, C, 4)
            bbox_target = bbox_target.transpose((1, 2, 0, 3))
            bbox_target = bbox_target.transpose((1, 2, 0, 3))

        return cls_target, bbox_target, bbox_mask
