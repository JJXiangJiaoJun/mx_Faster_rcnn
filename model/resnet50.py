from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
import gluoncv


class Bottleneck(gluon.Block):
    """
    ResNet中的基础结构 由1x1conv+3x3conv+1x1conv组成
    @:parameter
    planes : int
        输出的通道数
    downsample : gluon.Block    默认 None
        设置该结构是否要降采样，一般是每个小层的第一个Bottleneck设置降采样
    strides : int
        降采样的步长 conv3x3中设置
    norm_layer : gluon.Block
        是否要使用BN层
   
    """
    expansion = 4

    def __init__(self, planes, downsample=None, strides=1, norm_layer=None,
                 norm_kwargs={}, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        # conv1x1
        self.bn1 = norm_layer(**norm_kwargs)
        self.relu1 = nn.Activation('relu')
        self.conv1 = nn.Conv2D(channels=planes, kernel_size=1, strides=1, use_bias=False)
        # conv3x3
        self.bn2 = norm_layer(**norm_kwargs)
        self.relu2 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(channels=planes, kernel_size=3, strides=strides, padding=1, use_bias=False)
        # conv1x1
        self.bn3 = norm_layer(**norm_kwargs)
        self.relu3 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(channels=planes * 4, kernel_size=1, strides=1, use_bias=False)

        self.downsample = downsample
        self.strides = strides

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu3(out + residual)
        return out


class ResNetV1b(gluon.Block):
    """
    @:parameter
    block : gluon.Block 
        定义的残差小块
    layers : list of int
        定义每层中残差小块的个数
    norm_layer : 
    
    use_global_stats :
    
    name_prefix :
    
    
    """

    def __init__(self, block, layers, norm_layer=nn.BatchNorm, norm_kwargs={},
                 use_global_stats=False, name_prefix='', **kwargs):
        super(ResNetV1b, self).__init__(prefix=name_prefix)
        self.inplanes = 64

        self.norm_kwargs = norm_kwargs
        if use_global_stats:
            self.norm_kwargs['use_global_stats'] = True
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                   padding=3, use_bias=False)

            self.bn1 = norm_layer(**norm_kwargs)
            self.relu1 = nn.Activation('relu')

            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.layer1 = self._make_layer(1, block, 64, layers[0], strides=2, norm_layer=norm_layer)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, norm_layer=norm_layer)

            self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2, norm_layer=norm_layer)

    def _make_layer(self, stage_index, block, planes, blocks,
                    strides=1, norm_layer=None):
        """
        构建resnet的每一个小层
        @:parameter
        stage_index : int
            表示当前为第几个大层
        planes : int
            当前层最终的通道数
        layers : int
            当前层有多少个block   
        @:return: 
        """

        downsample = None
        # 表示要下采样
        if strides != 1:
            downsample = nn.Sequential(prefix='down%d_' % stage_index)
            with downsample.name_scope():
                downsample.add(nn.Conv2D(channels=planes * block.expansion, kernel_size=1,
                                         strides=strides, use_bias=False))
                downsample.add(norm_layer(**self.norm_kwargs))

        layers = nn.Sequential(prefix="layers{}_".format(stage_index))

        with layers.name_scope():
            # 第一个block进行下采样
            layers.add(block(planes, downsample=downsample, strides=strides,
                             norm_layer=norm_layer, norm_kwargs=self.norm_kwargs))

            self.inplanes = planes * 4

            for i in range(1, blocks):
                layers.add(block(planes, norm_layer=norm_layer, norm_kwargs=self.norm_kwargs))

        return layers

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


def get_resnet50v1b(pretrained=False, root='', **kwargs):
    model = ResNetV1b(Bottleneck, [3, 4, 6, 3], name_prefix='resnetv1b_', **kwargs)
    return model
