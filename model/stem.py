import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGStem(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.layers = self.make_layers()

    def make_layers(self):
        cfg = [64, 64, 'M', 128, 128, 'M', self.d_model, 'M']
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNetStem(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self.make_layer(PreActBlock, 64, 1, stride=1)
        self.layer2 = self.make_layer(PreActBlock, 128, 1, stride=2)
        self.bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, d_model, kernel_size=3,
                               stride=2, padding=1, bias=False)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.conv2(F.relu(self.bn(out)))
        return out


if __name__ == '__main__':
    m = VGGStem(d_model=512)
    x = torch.randn(1, 3, 32, 32)
    y = m(x)
    print(y.shape)

    m = ResNetStem(d_model=512)
    x = torch.randn(1, 3, 32, 32)
    y = m(x)
    print(y.shape)
