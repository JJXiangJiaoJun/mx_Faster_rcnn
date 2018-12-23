from mxnet import autograd
from mxnet.gluon import nn
import mxnet as mx
from .anchor import RPNAnchorGenerator
from .proposal import RPNProposal


# 定义RPN网络
# RPN网络输出应为一系列 region proposal  默认为 2000个
class RPN(nn.Block):
    """
    @输入参数
    channels : int
        卷积层的输出通道
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
        
    clip : float
        如果设置则将边界框剪切到该值
    nms_thresh : float
        非极大值抑制的阈值
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
    
    """

    def __init__(self, channels, stride, base_size, ratios,
                 scales, alloc_size, clip, nms_thresh,
                 train_pre_nms, train_post_nms, test_pre_nms, test_post_nms
                 , min_size, **kwargs):
        super(RPN, self).__init__(**kwargs)
        weight_initializer = mx.init.Normal(sigma=0.01)
        # 锚框生成器
        with self.name_scope():
            self.anchor_generator = RPNAnchorGenerator(stride, base_size, ratios, scales, alloc_size)
            anchor_depth = self.anchor_generator.num_depth
            self._rpn_proposal = RPNProposal(clip, nms_thresh, train_pre_nms,
                                             train_post_nms, test_pre_nms, test_post_nms, min_size,
                                             stds=(1., 1., 1., 1.))
            # 第一个提取特征的3x3卷积
            self.conv1 = nn.Sequential()
            self.conv1.add(
                nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, weight_initializer=weight_initializer),
                nn.Activation('relu'))
            # 预测偏移量和预测类别的卷积层
            # 使用sigmoid预测，减少通道数
            self.score = nn.Conv2D(anchor_depth, kernel_size=1, strides=1, padding=0,
                                   weight_initializer=weight_initializer)
            self.loc = nn.Conv2D(anchor_depth * 4, kernel_size=1, strides=1, padding=0,
                                 weight_initializer=weight_initializer)

    # 前向运算函数


    def forward(self, x, img):
        """
         产生锚框，并且对每个锚框进行二分类，以及回归预测
        ************************
         
         注意，这一阶段只是进行了粗采样，在RCNN中还要进行一次采样
         
         @:parameter
          -------------
          x : (B,C,H,W）
             由basenet提取出的特征图
         img : (B,C,H,W）
             图像tensor，用来剪切超出边框的锚框
    
         @:returns
         -----------------
         (1)训练阶段
         rpn_score : ndarray (B,train_post_nms,1)
             输出的region proposal 分数 (用来给RCNN采样)
    
         rpn_box : ndarray (B,train_post_nms,4)
             输出的region proposal坐标 Corner
    
         raw_score : ndarray (B,N,1)
             卷积层的原始输出，用来训练RPN
    
         rpn_bbox_pred : ndarray (B,N,4)
             卷积层的原始输出，用来训练RPN
    
         anchors : ndarray (1,N,4)
             生成的锚框
    
         (2)预测阶段
         
         rpn_score : ndarray (B,train_post_nms,1)
         输出的region proposal 分数 (用来给RCNN采样)
    
         rpn_box : ndarray (B,train_post_nms,4)
             输出的region proposal坐标 Corner
    
         """
        anchors = self.anchor_generator(x)
        # 提取特征
        feat = self.conv1(x)
        # 预测
        raw_rpn_score = self.score(feat)
        raw_rpn_score = raw_rpn_score.transpose((0, 2, 3, 1)).reshape(0, -1, 1)
        rpn_scores = mx.nd.sigmoid(mx.nd.stop_gradient(raw_rpn_score))
        raw_rpn_bbox = self.loc(feat)
        raw_rpn_bbox = raw_rpn_bbox.transpose((0, 2, 3, 1)).reshape(0, -1, 4)
        # 下面生成region proposal
        rpn_score, rpn_box = self._rpn_proposal(
            rpn_scores, mx.nd.stop_gradient(raw_rpn_bbox), anchors, img)
        # 处于训练阶段
        if autograd.is_training():
            # raw_score, rpn_bbox_pred 用于 RPN 的训练
            return rpn_score, rpn_box, raw_rpn_score, raw_rpn_bbox, anchors
        # 处于预测阶段
        return rpn_score, rpn_box
