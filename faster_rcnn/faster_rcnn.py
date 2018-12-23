from __future__ import absolute_import

import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import nn
from ..rcnn.rcnn import RCNN
from ..rpn.rpn import RPN
from .rcnn_target import RCNNTargetSampler, RCNNTargetGenerator
from ..model.resnet50 import get_resnet50v1b
from gluoncv.model_zoo import resnet50_v1b as gcv_resnet50_v1b


class FasterRCNN(RCNN):
    """
    @:parameter
    -------------
    """

    def __init__(self, features, top_features, classes,
                 short=600, max_size=1000, train_patterns=None,
                 nms_thresh=0.3, nms_topk=400, post_nms=100,
                 roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
                 rpn_channel=1024, base_size=16, scales=(8, 16, 32),
                 ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
                 rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                 rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                 num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=300,
                 **kwargs):

        super(FasterRCNN, self).__init__(
            features=features, top_features=top_features, classes=classes,
            short=short, max_size=max_size, train_patterns=train_patterns,
            nms_thresh=nms_thresh, nms_topk=nms_topk, post_nms=post_nms,
            roi_mode=roi_mode, roi_size=roi_size, stride=stride, clip=clip, **kwargs)

        self._max_batch = 1  # 最大支持batch=1
        self._num_sample = num_sample
        self._rpn_test_post_nms = rpn_test_post_nms
        self._target_generator = {RCNNTargetGenerator(self.num_class)}

        with self.name_scope():
            # Faster-RCNN的RPN
            self.rpn = RPN(
                channels=rpn_channel, stride=stride, base_size=base_size,
                scales=scales, ratios=ratios, alloc_size=alloc_size,
                clip=clip, nms_thresh=rpn_nms_thresh, train_pre_nms=rpn_train_pre_nms,
                train_post_nms=rpn_train_post_nms, test_pre_nms=rpn_test_pre_nms,
                test_post_nms=rpn_test_post_nms, min_size=rpn_min_size)

            # 用来给训练时Region Proposal采样，正负样本比例为0.25
            self.sampler = RCNNTargetSampler(
                num_image=self._max_batch, num_proposal=rpn_train_post_nms,
                num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
                pos_ratio=pos_ratio, max_num_gt=max_num_gt)
            # self.sampler = RCNNTargetSampler(
            #     num_images=self._max_batch, num_inputs=rpn_train_post_nms,
            #     num_samples=num_sample, pos_thresh=pos_iou_thresh,
            #     pos_ratio=pos_ratio, max_gt_box=max_num_gt)

    @property
    def target_generator(self):

        return list(self._target_generator)[0]

    def forward(self, x, gt_boxes=None):
        """
        :param x: ndarray (B,C,H,W)
        :return: 
        """

        def _split_box(x, num_outputs, axis, squeeze_axis=False):
            a = nd.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if not isinstance(a, (list, tuple)):
                return [a]
            return a

        # 首先用basenet抽取特征
        feat = self.features(x)

        # 输入RPN网络
        if autograd.is_training():
            # 训练过程
            img = nd.zeros_like(x)
            rpn_score, rpn_box, raw_rpn_score, raw_rpn_box, anchors = self.rpn(feat,img)
            # 采样输出
            rpn_box, samples, matches = self.sampler(rpn_box, rpn_score, gt_boxes)
        else:
            # 预测过程
            # output shape (B,N,4)
            _, rpn_box = self.rpn(feat, x)
        # 对输出的Region Proposal 进行采样
        # 输出送到后面运算的RoI
        # rois shape = (B,self._num_sampler,4),

        num_roi = self._num_sample if autograd.is_training() else self._rpn_test_post_nms

        # 将rois变为2D，加上batch_index
        with autograd.pause():
            roi_batchid = nd.arange(0, self._max_batch, repeat=num_roi,ctx=rpn_box.context)

            rpn_roi = nd.concat(*[roi_batchid.reshape((-1, 1)), rpn_box.reshape((-1, 4))], dim=-1)
            rpn_roi = nd.stop_gradient(rpn_roi)

        # RoI Pooling 层
        if self._roi_mode == 'pool':
            # (Batch*num_roi,channel,H,W)
            pool_feat = nd.ROIPooling(feat, rpn_roi, self._roi_size, 1 / self._stride)

        elif self._roi_mode == 'align':
            pool_feat = nd.contrib.ROIAlign(feat, rpn_roi, self._roi_size,
                                            1 / self._stride, sample_ratio=2)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._roi_mode))

        top_feat = self.top_features(pool_feat)
        avg_feat = self.global_avg_pool(top_feat)
        # 类别预测，回归预测
        # output shape (B*num_roi,(num_cls+1)) -> (B,N,C)
        cls_pred = self.class_predictor(avg_feat)
        # output shape (B*num_roi,(num_cls)*4) -> (B,N,C,4)
        box_pred = self.bbox_predictor(avg_feat)

        cls_pred = cls_pred.reshape((self._max_batch, num_roi, self.num_class + 1))
        box_pred = box_pred.reshape((self._max_batch, num_roi, self.num_class, 4))

        # 训练过程
        if autograd.is_training():

            return (cls_pred, box_pred, rpn_box, samples, matches,
                    raw_rpn_score, raw_rpn_box, anchors)
        # 预测过程
        # 还要进行的步骤，将预测的类别和预测的偏移量加到输入的RoI中
        else:
            # 直接输出所有类别的信息
            # cls_id (B,N,C) scores(B,N,C)
            cls_ids, scores = self.cls_decoder(nd.softmax(cls_pred, axis=-1))

            # 将所有的C调换到第一维
            # (B,N,C)  -----> (B,N,C,1) -------> (B,C,N,1)
            cls_ids = cls_ids.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
            # (B,N,C)  -----> (B,N,C,1) -------> (B,C,N,1)
            scores = scores.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
            # (B,N,C,4) -----> (B,C,N,4),
            box_pred = box_pred.transpose((0, 2, 1, 3))

            rpn_boxes = _split_box(rpn_box, num_outputs=self._max_batch, axis=0, squeeze_axis=False)
            cls_ids = _split_box(cls_ids, num_outputs=self._max_batch, axis=0, squeeze_axis=True)
            scores = _split_box(scores, num_outputs=self._max_batch, axis=0, squeeze_axis=True)
            box_preds = _split_box(box_pred, num_outputs=self._max_batch, axis=0, squeeze_axis=True)

            results = []
            # 对每个batch分别进行decoder nms
            for cls_id, score, box_pred, rpn_box in zip(cls_ids, scores, box_preds, rpn_boxes):
                # box_pred(C,N,4)   rpn_box(1,N,4)   box (C,N,4)
                box = self.box_decoder(box_pred, self.box_to_center(rpn_box))

                # cls_id (C,N,1) score (C,N,1) box (C,N,4)
                # result (C,N,6)
                res = nd.concat(*[cls_id, score, box], dim=-1)
                # nms操作 (C,self.nms_topk,6)
                res = nd.contrib.box_nms(res, overlap_thresh=self.nms_thresh, valid_thresh=0.0001,
                                         topk=self.nms_topk, coord_start=2, score_index=1, id_index=0,
                                         force_suppress=True)

                res = res.reshape((-3, 0))
                results.append(res)

            results = nd.stack(*results, axis=0)
            ids = nd.slice_axis(results, axis=-1, begin=0, end=1)
            scores = nd.slice_axis(results, axis=-1, begin=1, end=2)
            bboxes = nd.slice_axis(results, axis=-1, begin=2, end=6)

        # 输出为score,bbox
        return ids, scores, bboxes


def get_faster_rcnn(pretrained=False, ctx=mx.cpu(), root='', **kwargs):
    net = FasterRCNN(**kwargs)
    if pretrained:
        print("load parameter")
    return net


def get_faster_rcnn_resnet50v1b(pretraied=False, pretrained_base=True, **kwargs):
    base_network = get_resnet50v1b(pretrained=pretrained_base, **kwargs)
    features = nn.Sequential()
    top_features = nn.Sequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])

    return get_faster_rcnn(pretrained=pretraied, features=features, top_features=top_features, classes=10,
                           short=600, max_size=1000, train_patterns=train_patterns,
                           nms_thresh=0.3, nms_topk=400, post_nms=100,
                           roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
                           rpn_channel=1024, base_size=16, scales=(8, 16, 32),
                           ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
                           rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                           rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                           num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
                           **kwargs)


def faster_rcnn_resnet50_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""
    
    :param pretained:   bool or str
    :param pretrained_base: bool or str

    :return: model of faster-rcnn
    """
    from ..model.resnetv1b import resnet50_v1b
    # from ..data import VOCDetection
    classes = CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                         'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                         'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    pretrained_base = False if pretrained else pretrained_base
    base_network = gcv_resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                    use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])

    return get_faster_rcnn(pretrained=pretrained, features=features, top_features=top_features, classes=classes,
                           short=600, max_size=1000, train_patterns=train_patterns,
                           nms_thresh=0.3, nms_topk=400, post_nms=100,
                           roi_mode='align', roi_size=(14, 14), stride=16, clip=None,
                           rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
                           ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
                           rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                           rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                           num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
                           **kwargs)
