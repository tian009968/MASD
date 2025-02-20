'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np
from .aa import SPP111,SPP222,SPP333


__all__ = ['CIFAR_ResNet18', 'CIFAR_ResNet18_dks','CIFAR_ResNet18_ours', 'CIFAR_ResNet18_byot',
            'CIFAR_ResNet50', 'CIFAR_ResNet50_dks', 'CIFAR_ResNet50_byot',
            'manifold_mixup_CIFAR_ResNet18', 'manifold_mixup_CIFAR_ResNet50']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PreActBlock(nn.Module):
   
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


            
class CIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, branch_layers=[], is_bias=True):
        super(CIFAR_ResNet, self).__init__()
        self.branch_layers = branch_layers
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        inplanes_head0 = self.in_planes
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        inplanes_head2 = self.in_planes
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        inplanes_head1 = self.in_planes
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if len(self.branch_layers) != 0:
        
            
            self.spp222=SPP222()
            self.spp333=SPP333()
            
           
            
            
            self.in_planes = inplanes_head2
            self.layer3_head2 = self._make_layer(block, 256, branch_layers[0][0], stride=2)
            self.layer4_head2 = self._make_layer(block, 512, branch_layers[0][1], stride=2)
            self.fc_head2 = nn.Linear(512 * block.expansion, num_classes)

            self.in_planes = inplanes_head1
            self.layer4_head1 = self._make_layer(block, 512, branch_layers[1][0], stride=2)
            self.fc_head1 = nn.Linear(512 * block.expansion, num_classes, bias=is_bias)
            
            #self.fc_head0 = nn.Linear(512 * block.expansion, num_classes, bias=is_bias)
        


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, y=None, loss_type='cross_entropy', feature=False, embedding=False):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out1 = self.layer1(out)
        #print(out1.size())
        out2 = self.layer2(out1)
        #print(out2.size())
        out3 = self.layer3(out2)
        #print(out3.size())
        out = out4 = self.layer4(out3)
        f0 = out
        out = self.avgpool(out)
        embedding0 = out.view(out.size(0), -1)
        out = self.fc(embedding0)
        

        if len(self.branch_layers) != 0:
        
            
            out2=self.spp222(out2)
            x = self.layer3_head2(out2)
            x = self.layer4_head2(x)
            f2 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            x2 = self.fc_head2(x)
            
            out3=self.spp333(out3)
            x = self.layer4_head1(out3)
            f1 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            x1 = self.fc_head1(x)
       
        return embedding0


class ManifoldMixupCIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ManifoldMixupCIFAR_ResNet, self).__init__()
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        inplanes_head2 = self.in_planes
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        inplanes_head1 = self.in_planes
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def manifold_mixup(self, x, y, alpha=2.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y=None, alpha=2.0):
        index = random.randint(1,4)
        if self.training is False:
            index = -1
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        if index == 1:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer1(out)

        if index == 2:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer2(out)

        if index == 3:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer3(out)

        if index == 4:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer4(out)

        out = self.avgpool(out)
        embedding0 = out.view(out.size(0), -1)
        out = self.fc(embedding0)

        if self.training is False:
            return out
        else:
            return out, y_a, y_b, lam


def manifold_mixup_CIFAR_ResNet18(**kwargs):
    return ManifoldMixupCIFAR_ResNet(PreActBlock, [2,2,2,2], **kwargs)

def manifold_mixup_CIFAR_ResNet50(**kwargs):
    return ManifoldMixupCIFAR_ResNet(Bottleneck, [2,2,2,2], **kwargs)

def CIFAR_ResNet18(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[], **kwargs)

def CIFAR_ResNet18_dks(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_ours(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_byot(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 1], [1]], **kwargs)

def CIFAR_ResNet50(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[], **kwargs)

def CIFAR_ResNet50_dks(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[[1, 2], [2]], **kwargs)

def CIFAR_ResNet50_byot(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[[1, 1], [1]], **kwargs)


if __name__ == '__main__':
    net = CIFAR_ResNet18(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    import sys
    sys.path.append('..')
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))
'''
'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np
from .aa import SPP222,SPP333


__all__ = ['CIFAR_ResNet18', 'CIFAR_ResNet18_dks','CIFAR_ResNet18_dks8','CIFAR_ResNet18_dks9','CIFAR_ResNet18_dks10','CIFAR_ResNet18_dks6','CIFAR_ResNet18_dks7','CIFAR_ResNet18_dks1', 'CIFAR_ResNet18_dks2','CIFAR_ResNet18_dks3','CIFAR_ResNet18_dks4','CIFAR_ResNet18_dks5','CIFAR_ResNet18_byot',
            'CIFAR_ResNet50', 'CIFAR_ResNet50_dks','CIFAR_ResNet50_dks1', 'CIFAR_ResNet50_byot',
            'manifold_mixup_CIFAR_ResNet18', 'manifold_mixup_CIFAR_ResNet50']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PreActBlock(nn.Module):
    
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


            
class CIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, branch_layers=[], is_bias=True):
        super(CIFAR_ResNet, self).__init__()
        self.branch_layers = branch_layers
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        inplanes_head2 = self.in_planes
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        inplanes_head1 = self.in_planes
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if len(self.branch_layers) != 0:
            self.spp222=SPP222()
            self.spp333=SPP333()
            
            self.in_planes = inplanes_head2
            self.layer3_head2 = self._make_layer(block, 256, branch_layers[0][0], stride=2)
            self.layer4_head2 = self._make_layer(block, 512, branch_layers[0][1], stride=2)
            self.fc_head2 = nn.Linear(512 * block.expansion, num_classes)

            self.in_planes = inplanes_head1
            self.layer4_head1 = self._make_layer(block, 512, branch_layers[1][0], stride=2)
            self.fc_head1 = nn.Linear(512 * block.expansion, num_classes, bias=is_bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x,use_auxiliary=False, y=None, loss_type='cross_entropy', feature=False, embedding=False):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        
        out3 = self.layer3(out2)
        
                
        out = out4 = self.layer4(out3)
        f0 = out
        out = self.avgpool(out)
        embedding0 = out.view(out.size(0), -1)
        out = self.fc(embedding0)

        if len(self.branch_layers) != 0:
            
            if use_auxiliary:
                out2=self.spp222(out2)
            
            x = self.layer3_head2(out2)
            x = self.layer4_head2(x)
            f2 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            x2 = self.fc_head2(x)
            
            if use_auxiliary:
                out3=self.spp333(out3)
            
            x = self.layer4_head1(out3)
            f1 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            x1 = self.fc_head1(x)
        return embedding0


class ManifoldMixupCIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ManifoldMixupCIFAR_ResNet, self).__init__()
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        inplanes_head2 = self.in_planes
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        inplanes_head1 = self.in_planes
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def manifold_mixup(self, x, y, alpha=2.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y=None, alpha=2.0):
        index = random.randint(1,4)
        if self.training is False:
            index = -1
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        if index == 1:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer1(out)

        if index == 2:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer2(out)

        if index == 3:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer3(out)

        if index == 4:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer4(out)

        out = self.avgpool(out)
        embedding0 = out.view(out.size(0), -1)
        out = self.fc(embedding0)

        if self.training is False:
            return out
        else:
            return out, y_a, y_b, lam


def manifold_mixup_CIFAR_ResNet18(**kwargs):
    return ManifoldMixupCIFAR_ResNet(PreActBlock, [2,2,2,2], **kwargs)

def manifold_mixup_CIFAR_ResNet50(**kwargs):
    return ManifoldMixupCIFAR_ResNet(Bottleneck, [2,2,2,2], **kwargs)

def CIFAR_ResNet18(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[], **kwargs)

def CIFAR_ResNet18_dks(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dks1(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dks2(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dks3(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dks4(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dks5(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dks6(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[2, 2], [2]], **kwargs)

def CIFAR_ResNet18_dks7(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dks8(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dks9(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[2, 1], [2]], **kwargs)

def CIFAR_ResNet18_dks10(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)

def CIFAR_ResNet18_byot(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 1], [1]], **kwargs)

def CIFAR_ResNet50(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[], **kwargs)

def CIFAR_ResNet50_dks(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet50_dks1(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[[1, 2], [2]], **kwargs)

def CIFAR_ResNet50_byot(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[[1, 1], [1]], **kwargs)


if __name__ == '__main__':
    net = CIFAR_ResNet18(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    import sys
    sys.path.append('..')
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))
'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np
from .aa import SPP222,SPP333

__all__ = ['CIFAR_ResNet18', 'CIFAR_ResNet18_dks', 'CIFAR_ResNet18_dksours','CIFAR_ResNet18_dks2','CIFAR_ResNet18_dks3','CIFAR_ResNet18_byot',
            'CIFAR_ResNet50', 'CIFAR_ResNet50_dks', 'CIFAR_ResNet50_byot',
            'manifold_mixup_CIFAR_ResNet18', 'manifold_mixup_CIFAR_ResNet50','ResNet18','ResNet18_byot','ResNet18_ours1','ResNet18_ours','ResNet18_ours2','ResNet18_ours3']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, branch_layers=[],num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        self.branch_layers = branch_layers
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        inplanes_head2 = self.inplanes
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        inplanes_head1 = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)
     
        if len(self.branch_layers) != 0:
            
            self.spp222=SPP222()
            self.spp333=SPP333()
            
      
            self.inplanes = inplanes_head2
            self.layer3_head2 = self._make_layer(block, 256, branch_layers[0][0], stride=2, norm_layer=norm_layer)
            self.layer4_head2 = self._make_layer(block, 512, branch_layers[0][1], stride=2, norm_layer=norm_layer)
            self.fc_head2 = nn.Linear(512 * block.expansion, num_classes)

            self.inplanes = inplanes_head1
            self.layer4_head1 = self._make_layer(block, 512, branch_layers[1][0], stride=2, norm_layer=norm_layer)
            self.fc_head1 = nn.Linear(512 * block.expansion, num_classes)
      
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
  

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)
  
    
    def forward(self, x, use_auxiliary=False, y=None, loss_type='cross_entropy', feature=False, embedding=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = out4 = self.layer4(out3)
        f0 = out

        out = self.avgpool(out)
        embedding0 = out.view(out.size(0), -1)
        out = self.fc(embedding0)
        
        if len(self.branch_layers) != 0:
            
            if use_auxiliary:
                out2=self.spp222(out2)
            
            x = self.layer3_head2(out2)
            x = self.layer4_head2(x)
            f2 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            x2 = self.fc_head2(x)
            
            if use_auxiliary:
                out3=self.spp333(out3)
            
            x = self.layer4_head1(out3)
            f1 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            x1 = self.fc_head1(x)
        
            if feature:
                return [out, x1, x2], [embedding0, f0, f1, f2]
            else:
                
                return [out, x1, x2]
        else:
            if loss_type == 'cross_entropy':
                if feature:
                    return out, [out1, out2, out3, out4]
                if embedding:
                    return out, embedding0
                else:
                    return out
            elif loss_type == 'virtual_softmax':
                target_w = self.fc.weight[y]
                L2_target_w = target_w.pow(2).sum(1, keepdim=True).pow(1. / 2.)
                x_target_w = embedding0.pow(2).sum(1, keepdim=True).pow(1. / 2.)
                out = torch.cat([out, L2_target_w * x_target_w], dim=1)
                return out

            
class CIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, branch_layers=[], is_bias=True):
        super(CIFAR_ResNet, self).__init__()
        self.branch_layers = branch_layers
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        inplanes_head2 = self.in_planes
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        inplanes_head1 = self.in_planes
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if len(self.branch_layers) != 0:
            self.spp222=SPP222()
            self.spp333=SPP333()
            self.in_planes = inplanes_head2
            self.layer3_head2 = self._make_layer(block, 256, branch_layers[0][0], stride=2)
            self.layer4_head2 = self._make_layer(block, 512, branch_layers[0][1], stride=2)
            self.fc_head2 = nn.Linear(512 * block.expansion, num_classes)

            self.in_planes = inplanes_head1
            self.layer4_head1 = self._make_layer(block, 512, branch_layers[1][0], stride=2)
            self.fc_head1 = nn.Linear(512 * block.expansion, num_classes, bias=is_bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, use_auxiliary=False, y=None, loss_type='cross_entropy', feature=False, embedding=False):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out = out4 = self.layer4(out3)
        f0 = out
        out = self.avgpool(out)
        embedding0 = out.view(out.size(0), -1)
        out = self.fc(embedding0)

        if len(self.branch_layers) != 0:
            if use_auxiliary:
                out2=self.spp222(out2)
            x = self.layer3_head2(out2)
            x = self.layer4_head2(x)
            f2 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            x2 = self.fc_head2(x)
            
            if use_auxiliary:
                out3=self.spp333(out3)

            x = self.layer4_head1(out3)
            f1 = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            x1 = self.fc_head1(x)
            if feature:
                return [out, x1, x2], [embedding0, f0, f1, f2]
            else:
                return [out, x1, x2]
        else:
            if loss_type == 'cross_entropy':
                if feature:
                    return out, [out1, out2, out3, out4]
                if embedding:
                    return out, embedding0
                else:
                    return out
            elif loss_type == 'virtual_softmax':
                target_w = self.fc.weight[y]
                L2_target_w = target_w.pow(2).sum(1, keepdim=True).pow(1. / 2.)
                x_target_w = embedding0.pow(2).sum(1, keepdim=True).pow(1. / 2.)
                out = torch.cat([out, L2_target_w * x_target_w], dim=1)
                return out
        #return embedding0


class ManifoldMixupCIFAR_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ManifoldMixupCIFAR_ResNet, self).__init__()
        self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        inplanes_head2 = self.in_planes
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        inplanes_head1 = self.in_planes
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def manifold_mixup(self, x, y, alpha=2.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y=None, alpha=2.0):
        index = random.randint(1,4)
        if self.training is False:
            index = -1
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        if index == 1:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer1(out)

        if index == 2:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer2(out)

        if index == 3:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer3(out)

        if index == 4:
            mixed_x, y_a, y_b, lam = self.manifold_mixup(out, y)
            out = mixed_x

        out = self.layer4(out)

        out = self.avgpool(out)
        embedding0 = out.view(out.size(0), -1)
        out = self.fc(embedding0)

        if self.training is False:
            return out
        else:
            return out, y_a, y_b, lam



def ResNet18(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], branch_layers=[], **kwargs)
    
def ResNet18_dks(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def ResNet18_byot(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], branch_layers=[[1, 1], [1]], **kwargs)
 
def ResNet18_ours(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], branch_layers=[[2, 2], [2]], **kwargs)
      
def ResNet18_ours1(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], branch_layers=[[2, 2], [2]], **kwargs) 
      
def ResNet18_ours2(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], branch_layers=[[2, 2], [2]], **kwargs)
    
def ResNet18_ours3(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], branch_layers=[[2, 2], [2]], **kwargs)


def manifold_mixup_CIFAR_ResNet18(**kwargs):
    return ManifoldMixupCIFAR_ResNet(PreActBlock, [2,2,2,2], **kwargs)

def manifold_mixup_CIFAR_ResNet50(**kwargs):
    return ManifoldMixupCIFAR_ResNet(Bottleneck, [2,2,2,2], **kwargs)

def CIFAR_ResNet18(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[], **kwargs)

def CIFAR_ResNet18_dks(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dksours(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[2, 2], [2]], **kwargs)

def CIFAR_ResNet18_dks2(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[2, 2], [2]], **kwargs)
    
def CIFAR_ResNet18_dks3(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[2, 2], [2]], **kwargs)


def CIFAR_ResNet18_byot(pretrained=False, **kwargs):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], branch_layers=[[1, 1], [1]], **kwargs)

def CIFAR_ResNet50(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[], **kwargs)

def CIFAR_ResNet50_dks(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[[1, 2], [2]], **kwargs)

def CIFAR_ResNet50_byot(pretrained=False, **kwargs):
    return CIFAR_ResNet(Bottleneck, [3,4,6,3], branch_layers=[[1, 1], [1]], **kwargs)


if __name__ == '__main__':
    net = CIFAR_ResNet18(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    import sys
    sys.path.append('..')
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))