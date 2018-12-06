import torch
import torch.nn as nn


class ResNet23(nn.Module):
    def __init__(self):
        super(ResNet23, self).__init__()
        self.features = ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True])
        self.feature_size = 512

    def forward(self, x):
        x = self.features(x)
        return x


# -------------
# ResNet tools
# -------------
# ResNet blocks from torchvision

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def center_crop(x):
    """
    center crop layer. crop [1:-2] to eliminate padding influence.
    input x can be a Variable or Tensor
    """
    return x[:, :, 1:-1, 1:-1].contiguous()


def center_crop_conv7(x):
    """
    center crop layer. crop [2:-3] to eliminate padding influence.
    input x can be a Variable or Tensor
    """
    return x[:, :, 2:-2, 2:-2].contiguous()


class Bottleneck_CI(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None):
        super(Bottleneck_CI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        
        if self.last_relu:    # feature out no relu
            out = self.relu(out)

        out = center_crop(out)   # in-layer crop

        return out


class ResNet(nn.Module):
    """
    last_relus:　a list controls whether use relu in the end of each stage. eg. [True, True, True, False]
    s2p_flags:　 a list controls whether use strides2pool in  each stage. eg. [True, True, False, False]
    Res23: ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True])
    """
    def __init__(self, block, layers, last_relus, s2p_flags):
        self.inplanes = 64
        self.stage_len = len(layers)
        print('we will use {} stages of resnet50.'.format(self.stage_len))
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(block, 64, layers[0], stride2pool=s2p_flags[0], last_relu=last_relus[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride2pool=s2p_flags[1], last_relu=last_relus[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, last_relu, stride=1, stride2pool=False):
        """
        :param block:
        :param planes:
        :param blocks:
        :param stride:
        :param stride2pool: translate (3,2) conv to (3, 1)conv + (2, 2)pool
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, last_relu=True, stride=stride, downsample=downsample))
        if stride2pool:
            layers.append(self.maxpool)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks-1:
                layers.append(block(self.inplanes, planes, last_relu=last_relu))
            else:
                layers.append(block(self.inplanes, planes, last_relu=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)          # stride = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = center_crop_conv7(x)
        x = self.maxpool(x)        # stride = 4

        x = self.layer1(x)
        x = self.layer2(x)         # stride = 8

        return x


if __name__ == '__main__':
    # check model
    net = ResNet23()
    net.cuda()

    from torch.autograd import Variable

    search = torch.rand(1, 3, 127, 127)
    search = Variable(search).cuda()
    out = net(search)

    print('net structure: ')
    print(net)
    print('template feature output size: ')
    print(out.size())

    print('check done!')
