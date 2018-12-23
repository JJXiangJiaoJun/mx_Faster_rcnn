from __future__ import absolute_import

import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd
import numpy as np
from ..utils.coder import NormalizedPerClassBoxCenterEncoder, MultiClassEncoder


class RCNNTargetSampler(gluon.Block):
    """A sampler to choose positive/negative samples from RCNN Proposals

    Parameters
    ----------
    num_image: int
        Number of input images.
    num_proposal: int
        Number of input proposals.
    num_sample : int
        Number of samples for RCNN targets.
    pos_iou_thresh : float
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
        Proposal whose IOU smaller than ``pos_iou_thresh`` is regarded as negative samples.
    pos_ratio : float
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    max_num_gt : int
        Maximum ground-truth number in whole training dataset. This is only an upper bound, not
        necessarily very precise. However, using a very big number may impact the training speed.

    """

    def __init__(self, num_image, num_proposal, num_sample, pos_iou_thresh, pos_ratio, max_num_gt):
        super(RCNNTargetSampler, self).__init__()
        self._num_image = num_image
        self._num_proposal = num_proposal
        self._num_sample = num_sample
        self._max_pos = int(round(num_sample * pos_ratio))
        self._pos_iou_thresh = pos_iou_thresh
        self._max_num_gt = max_num_gt

    # pylint: disable=arguments-differ
    def forward(self, rois, scores, gt_boxes):
        """Handle B=self._num_image by a for loop.

        Parameters
        ----------
        rois: (B, self._num_input, 4) encoded in (x1, y1, x2, y2).
        scores: (B, self._num_input, 1), value range [0, 1] with ignore value -1.
        gt_boxes: (B, M, 4) encoded in (x1, y1, x2, y2), invalid box should have area of 0.

        Returns
        -------
        rois: (B, self._num_sample, 4), randomly drawn from proposals
        samples: (B, self._num_sample), value +1: positive / 0: ignore / -1: negative.
        matches: (B, self._num_sample), value between [0, M)

        """
        F = nd
        with autograd.pause():
            # collect results into list
            new_rois = []
            new_samples = []
            new_matches = []
            for i in range(self._num_image):
                roi = F.squeeze(F.slice_axis(rois, axis=0, begin=i, end=i + 1), axis=0)
                score = F.squeeze(F.slice_axis(scores, axis=0, begin=i, end=i + 1), axis=0)
                gt_box = F.squeeze(F.slice_axis(gt_boxes, axis=0, begin=i, end=i + 1), axis=0)
                gt_score = F.ones_like(F.sum(gt_box, axis=-1, keepdims=True))

                # concat rpn roi with ground truth
                all_roi = F.concat(roi, gt_box, dim=0)
                all_score = F.concat(score, gt_score, dim=0).squeeze(axis=-1)
                # calculate (N, M) ious between (N, 4) anchors and (M, 4) bbox ground-truths
                # cannot do batch op, will get (B, N, B, M) ious
                ious = F.contrib.box_iou(all_roi, gt_box, format='corner')
                # match to argmax iou
                ious_max = ious.max(axis=-1)
                ious_argmax = ious.argmax(axis=-1)
                # init with 2, which are neg samples
                mask = F.ones_like(ious_max) * 2
                # mark all ignore to 0
                mask = F.where(all_score < 0, F.zeros_like(mask), mask)
                # mark positive samples with 3
                pos_mask = ious_max >= self._pos_iou_thresh
                mask = F.where(pos_mask, F.ones_like(mask) * 3, mask)

                # shuffle mask
                rand = F.random.uniform(0, 1, shape=(self._num_proposal + self._max_num_gt,),ctx=ious_argmax.context)
                rand = F.slice_like(rand, ious_argmax)
                index = F.argsort(rand)
                mask = F.take(mask, index)
                ious_argmax = F.take(ious_argmax, index)

                # sample pos samples
                order = F.argsort(mask, is_ascend=False)
                topk = F.slice_axis(order, axis=0, begin=0, end=self._max_pos)
                topk_indices = F.take(index, topk)
                topk_samples = F.take(mask, topk)
                topk_matches = F.take(ious_argmax, topk)
                # reset output: 3 pos 2 neg 0 ignore -> 1 pos -1 neg 0 ignore
                topk_samples = F.where(topk_samples == 3,
                                       F.ones_like(topk_samples), topk_samples)
                topk_samples = F.where(topk_samples == 2,
                                       F.ones_like(topk_samples) * -1, topk_samples)

                # sample neg samples
                index = F.slice_axis(index, axis=0, begin=self._max_pos, end=None)
                mask = F.slice_axis(mask, axis=0, begin=self._max_pos, end=None)
                ious_argmax = F.slice_axis(ious_argmax, axis=0, begin=self._max_pos, end=None)
                # change mask: 4 neg 3 pos 0 ignore
                mask = F.where(mask == 2, F.ones_like(mask) * 4, mask)
                order = F.argsort(mask, is_ascend=False)
                num_neg = self._num_sample - self._max_pos
                bottomk = F.slice_axis(order, axis=0, begin=0, end=num_neg)
                bottomk_indices = F.take(index, bottomk)
                bottomk_samples = F.take(mask, bottomk)
                bottomk_matches = F.take(ious_argmax, bottomk)
                # reset output: 4 neg 3 pos 0 ignore -> 1 pos -1 neg 0 ignore
                bottomk_samples = F.where(bottomk_samples == 3,
                                          F.ones_like(bottomk_samples), bottomk_samples)
                bottomk_samples = F.where(bottomk_samples == 4,
                                          F.ones_like(bottomk_samples) * -1, bottomk_samples)

                # output
                indices = F.concat(topk_indices, bottomk_indices, dim=0)
                samples = F.concat(topk_samples, bottomk_samples, dim=0)
                matches = F.concat(topk_matches, bottomk_matches, dim=0)

                new_rois.append(all_roi.take(indices))
                new_samples.append(samples)
                new_matches.append(matches)
            # stack all samples together
            new_rois = F.stack(*new_rois, axis=0)
            new_samples = F.stack(*new_samples, axis=0)
            new_matches = F.stack(*new_matches, axis=0)
        return new_rois, new_samples, new_matches


class RCNNTargetGenerator(gluon.Block):
    """RCNN target encoder to generate matching target and regression target values.

    Parameters
    ----------
    num_class : int
        Number of total number of positive classes.
    means : iterable of float, default is (0., 0., 0., 0.)
        Mean values to be subtracted from regression targets.
    stds : iterable of float, default is (.1, .1, .2, .2)
        Standard deviations to be divided from regression targets.

    """

    def __init__(self, num_class, means=(0., 0., 0., 0.), stds=(.1, .1, .2, .2)):
        super(RCNNTargetGenerator, self).__init__()
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedPerClassBoxCenterEncoder(
            num_class=num_class, means=means, stds=stds)

    # pylint: disable=arguments-differ
    def forward(self, roi, samples, matches, gt_label, gt_box):
        """Components can handle batch images

        Parameters
        ----------
        roi: (B, N, 4), input proposals
        samples: (B, N), value +1: positive / -1: negative.
        matches: (B, N), value [0, M), index to gt_label and gt_box.
        gt_label: (B, M), value [0, num_class), excluding background class.
        gt_box: (B, M, 4), input ground truth box corner coordinates.

        Returns
        -------
        cls_target: (B, N), value [0, num_class + 1), including background.
        box_target: (B, N, C, 4), only foreground class has nonzero target.
        box_weight: (B, N, C, 4), only foreground class has nonzero weight.

        """
        with autograd.pause():
            # cls_target (B, N)
            cls_target = self._cls_encoder(samples, matches, gt_label)
            # box_target, box_weight (C, B, N, 4)
            box_target, box_mask = self._box_encoder(
                samples, matches, roi, gt_label, gt_box)
            # modify shapes to match predictions
            # box (C, B, N, 4) -> (B, N, C, 4)
            box_target = box_target.transpose((1, 2, 0, 3))
            box_mask = box_mask.transpose((1, 2, 0, 3))
        return cls_target, box_target, box_mask

# class RCNNTargetSampler(gluon.Block):
#     """
#     @:parameter
#     ------------
#     num_images : int
#         每个batch的图片数，目前仅支持1
#     num_inputs : int
#         输入的RoI 数量
#     num_samples : int
#         输出的采样 RoI 数量
#     pos_thresh : float
#         正类样本阈值

#     pos_ratio : float
#         采样正样本的比例

#     max_gt_box : int


#     """

#     def __init__(self, num_images, num_inputs, num_samples, pos_thresh, pos_ratio, max_gt_box, **kwargs):
#         super(RCNNTargetSampler, self).__init__(**kwargs)
#         self._num_images = num_images
#         self._num_inputs = num_inputs
#         self._num_samples = num_samples
#         self._pos_thresh = pos_thresh
#         self._pos_ratios = pos_ratio
#         self._max_pos = int(np.round(num_samples * pos_ratio))
#         self._max_gt_box = max_gt_box

#     def forward(self, rois, scores, gt_bboxes):
#         """
#         @:parameter
#         -----------
#         rois : ndarray (B,self._num_inputs,4)
#             RPN输出的roi区域坐标，Corner

#         scores : ndarray (B,self._num_inputs,1)
#             RPN输出的roi区域分数，(0,1) -1表示忽略

#         gt_bboxes:ndarray (B,M,4)
#             ground truth box 坐标

#         @:returns
#         -----------
#         new_rois : ndarray (B,self._num_samples,4)
#             采样后的RoI区域
#         new_samples : ndarray (B,self._num_samples,1)
#             采样后RoI区域的标签 1:pos -1:neg 0:ignore
#         new_matches : ndarray (B,self._num_samples,1)
#             采样后的RoI匹配的锚框编号 [0,M)

#         """

#         new_rois, new_samples, new_matches = [], [], []

#         # 对每个batch分别进行处理
#         for i in range(self._num_images):
#             roi = nd.squeeze(nd.slice_axis(rois, axis=0, begin=i, end=i + 1), axis=0)
#             score = nd.squeeze(nd.slice_axis(scores, axis=0, begin=i, end=i + 1), axis=0)
#             gt_bbox = nd.squeeze(nd.slice_axis(gt_bboxes, axis=0, begin=i, end=i + 1), axis=0)

#             # 将ground truth的分数设置为1 形状为(M,1)
#             gt_score = nd.ones_like(nd.sum(gt_bbox, axis=-1, keepdims=True))

#             # 将ground truth 和 roi 拼接 (N+M,4) (N+m,1)
#             roi = nd.concat(roi, gt_bbox, dim=0)
#             score = nd.concat(score, gt_score, dim=0).squeeze(axis=-1)

#             # 计算iou   (N+M,M)
#             iou = nd.contrib.box_iou(roi, gt_bbox, format='corner')
#             # (N+M,)
#             iou_max = nd.max(iou, axis=-1)
#             # (N+M,)  与哪个ground truth 匹配
#             iou_argmax = nd.argmax(iou, axis=-1)

#             # 将所有的标记为 2 neg
#             mask = nd.ones_like(iou_argmax) * 2
#             # 标记ignore 为 0
#             mask = nd.where(score < 0, nd.zeros_like(mask), mask)

#             # 将正类标记为 3 pos
#             pos_idx = (iou_max >= self._pos_thresh)

#             mask = nd.where(pos_idx, nd.ones_like(mask) * 3, mask)

#             # 下面进行shuffle操作
#             rand = nd.random.uniform(0, 1, shape=(self._num_inputs + self._max_gt_box,))
#             # 取前面 N+M 个 对mask 做shuffle操作
#             rand = nd.slice_like(rand, iou_argmax)
#             # shuffle 操作后的 index
#             index = nd.argsort(rand)
#             # 将三个结果进行shuffle
#             mask = nd.take(mask, index)
#             iou_argmax = nd.take(iou_argmax, index)

#             # 下面进行采样
#             # 排序 3:pos 2:neg 0:ignore
#             order = nd.argsort(mask, is_ascend=False)
#             # 取topk个作为正例
#             topk = nd.slice_axis(order, axis=0, begin=0, end=self._max_pos)
#             # 下面取出相对应的值
#             pos_indices = nd.take(index, topk)
#             pos_samples = nd.take(mask, topk)
#             pos_matches = nd.take(iou_argmax, topk)

#             # 下面将原来的标签改了
#             pos_samples = nd.where(pos_samples == 3, nd.ones_like(pos_samples), pos_samples)
#             pos_samples = nd.where(pos_samples == 2, nd.ones_like(pos_samples) * -1, pos_samples)

#             index = nd.slice_axis(index, axis=0, begin=self._max_pos, end=None)
#             mask = nd.slice_axis(mask, axis=0, begin=self._max_pos, end=None)
#             iou_argmax = nd.slice_axis(iou_argmax, axis=0, begin=self._max_pos, end=None)

#             # 对负样本进行采样
#             # neg 2---->4
#             mask = nd.where(mask == 2, nd.ones_like(mask) * 4, mask)
#             order = nd.argsort(mask, is_ascend=False)
#             num_neg = self._num_samples - self._max_pos
#             bottomk = nd.slice_axis(order, axis=0, begin=0, end=num_neg)

#             neg_indices = nd.take(index, bottomk)
#             neg_samples = nd.take(mask, bottomk)
#             neg_matches = nd.take(iou_argmax, bottomk)

#             neg_samples = nd.where(neg_samples == 3, nd.ones_like(neg_samples), neg_samples)
#             neg_samples = nd.where(neg_samples == 4, nd.ones_like(neg_samples) * -1, neg_samples)

#             # 输出
#             new_idx = nd.concat(pos_indices, neg_indices, dim=0)
#             new_sample = nd.concat(pos_samples, neg_samples, dim=0)
#             new_match = nd.concat(pos_matches, neg_matches, dim=0)

#             new_rois.append(roi.take(new_idx))
#             new_samples.append(new_sample)
#             new_matches.append(new_match)

#         new_rois = nd.stack(*new_rois, axis=0)
#         new_samples = nd.stack(*new_samples, axis=0)
#         new_matches = nd.stack(*new_matches, axis=0)

#         return new_rois, new_samples, new_matches


# class RCNNTargetGenerator(gluon.Block):
#     """
#     @:parameter
#     -----------
#     num_class : int
#         总类别数，不包括背景
#     means : tuple of float

#     stds :  tuple of float



#     """

#     def __init__(self, num_class, means=(0., 0., 0., 0.), stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
#         super(RCNNTargetGenerator, self).__init__(**kwargs)
#         self._num_class = num_class
#         self._bbox_encoder = NormalizedPerClassBoxCenterEncoder(num_class, stds, means)
#         self._cls_encoder = MultiClassEncoder()

#     def forward(self, rois, samples, matches, gt_labels, gt_boxes):
#         """
#         :param rois: (B,N,4)
#             采样之后的roi坐标
#         :param samples: (B,N,1)
#             采样之后的标签 +1:pos / -1:neg  0:ignore
#         :param matches:
#             匹配的ground truth索引 [0，M)
#         :param gt_boxes: (B,M,4)
#         :param gt_labels: (B,M,)


#         :return:
#         cls_target : (B,N)
#             [0，num_class+1)类别标签，包括背景
#         box_target:  (B,N,C,4)

#         bbox_mask : (B,N,C,4)

#         """
#         with autograd.pause():
#             # 首先进行采样

#             # 对类别编码
#             cls_target = self._cls_encoder(samples, matches, gt_labels)
#             # 对bbox进行偏移量标注
#             bbox_target, bbox_mask = self._bbox_encoder(samples, matches, rois, gt_labels, gt_boxes)

#             # modify shapes to match predictions
#             # box (C, B, N, 4) -> (B, N, C, 4)
#             bbox_target = bbox_target.transpose((1, 2, 0, 3))
#             bbox_mask = bbox_mask.transpose((1, 2, 0, 3))

#         return cls_target, bbox_target, bbox_mask
