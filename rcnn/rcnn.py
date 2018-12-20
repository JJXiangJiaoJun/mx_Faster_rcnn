from __future__ import absolute_import

from mxnet import gluon
import mxnet as mx
from mxnet.gluon import nn
from ..utils.bbox import BBoxCornerToCenter
from ..utils.coder import NormalizedBoxCenterDecoder, MultiPerClassDecoder


class RCNN(gluon.Block):
    """
    @:parameter
    ------------
    features : gluon.block
        Base Feature的提取网络，在RoI pooling层之前的
    top_features : gluon.block
        RoI pooling层之后的特征提取卷积层
    classes : iterable of str
        classes 的名字，长度为'num_cls'
    short : int
        输入图像较短边的尺寸
    max_size : int
        输入图像较长边的最大尺寸
    train_patterns : str
        匹配需要训练的参数层
    num_thresh : float
        非极大值抑制的阈值
    nms_topk : int
    
    
    """

    def __init__(self, features, top_features, classes,
                 short, max_size, train_patterns,
                 nms_thresh, nms_topk, post_nms,
                 roi_mode, roi_size, stride, clip, **kwargs):
        super(RCNN, self).__init__(**kwargs)
        self.classes = classes
        self.num_class = len(classes)
        self.short = short
        self.max_size = max_size
        self.train_patterns = train_patterns
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        assert self.num_class > 0, "Invalid number of class : {}".format(self.num_class)
        assert roi_mode.lower() in ['align', 'pool'], "Invalid roi_mode: {}".format(roi_mode)
        self._roi_mode = roi_mode.lower()
        assert len(roi_size) == 2, "Require (h, w) as roi_size, given {}".format(roi_size)
        self._roi_size = roi_size
        self._stride = stride

        with self.name_scope():
            self.features = features
            self.top_features = top_features
            self.global_avg_pool = nn.GlobalAvgPool2D()
            self.class_predictor = nn.Dense(
                self.num_class + 1, weight_initializer=mx.init.Normal(0.01)
            )
            self.bbox_predictor = nn.Dense(
                self.num_class * 4, weight_initializer=mx.init.Normal(0.001)
            )
            self.cls_decoder = MultiPerClassDecoder(num_class=self.num_class + 1)
            self.box_to_center = BBoxCornerToCenter()
            self.box_decoder = NormalizedBoxCenterDecoder(clip=clip)

    def collect_train_params(self, select=None):
        """Collect trainable params.

        This function serves as a help utility function to return only
        trainable parameters if predefined by experienced developer/researcher.
        For example, if cross-device BatchNorm is not enabled, we will definitely
        want to fix BatchNorm statistics to avoid scaling problem because RCNN training
        batch size is usually very small.

        Parameters
        ----------
        select : select : str
            Regular expressions for parameter match pattern

        Returns
        -------
        The selected :py:class:`mxnet.gluon.ParameterDict`

        """
        if select is None:
            return self.collect_params(self.train_patterns)
        return self.collect_params(select)

    def set_nms(self, nms_thresh=0.3, nms_topk=400, post_nms=100):
        """Set NMS parameters to the network.

        .. Note::
            If you are using hybrid mode, make sure you re-hybridize after calling
            ``set_nms``.

        Parameters
        ----------
        nms_thresh : float, default is 0.3.
            Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def reset_class(self, classes):
        """Reset class categories and class predictors.

        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.

        """
        self._clear_cached_op()
        self.classes = classes
        self.num_class = len(classes)
        with self.name_scope():
            self.class_predictor = nn.Dense(
                self.num_class + 1, weight_initializer=mx.init.Normal(0.01),
                prefix=self.class_predictor.prefix)
            self.box_predictor = nn.Dense(
                self.num_class * 4, weight_initializer=mx.init.Normal(0.001),
                prefix=self.box_predictor.prefix)
            self.cls_decoder = MultiPerClassDecoder(num_class=self.num_class + 1)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, width, height):
        """Not implemented yet."""
        raise NotImplementedError
