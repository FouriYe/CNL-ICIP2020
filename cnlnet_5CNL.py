# --------------------------------------------------------
# Cross-layer Non-Local Network
# Copyright (c) 2020 Zihan Ye
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import math
import torch
import torch.nn as nn

from termcolor import cprint
from collections import OrderedDict
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np

unloader = transforms.ToPILImage()

def model_hub(arch, pretrained=True, pool_size=7):
    """Model hub.
    """
    if arch == '50':
        return cnlnet50(arch=arch,
                        pretrained=pretrained,
                        pool_size=pool_size)
    elif arch == '101':
        return cnlnet101(arch=arch,
                         pretrained=pretrained,
                         pool_size=pool_size)
    else:
        raise NameError("The arch '{}' is not supported yet in this repo. \
                You can add it by yourself.".format(arch))

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out

class CNL_5(nn.Module):
    def __init__(self, response_inplanes, perspective_inplanes, planes,output_planes, use_scale=False):
        self.use_scale = use_scale
        super(CNL_5, self).__init__()
        
        self.softmax = nn.Softmax(dim=2)
        self.t = nn.Conv2d(perspective_inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.z = nn.Conv2d(planes, perspective_inplanes, kernel_size=1, stride=1, bias=False)
        
        self.p4 = nn.Conv2d(response_inplanes[4], planes, kernel_size=1, stride=1, bias=False)
        self.g4 = nn.Conv2d(response_inplanes[4], planes, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(perspective_inplanes)
        
        self.p3 = nn.Conv2d(response_inplanes[3], planes, kernel_size=1, stride=1, bias=False)
        self.g3 = nn.Conv2d(response_inplanes[3], planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(perspective_inplanes)
        
        self.p2 = nn.Conv2d(response_inplanes[2], planes, kernel_size=1, stride=1, bias=False)
        self.g2 = nn.Conv2d(response_inplanes[2], planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(perspective_inplanes)
        
        self.p1 = nn.Conv2d(response_inplanes[1], planes, kernel_size=1, stride=1, bias=False)
        self.g1 = nn.Conv2d(response_inplanes[1], planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(perspective_inplanes)
        
        self.p0 = nn.Conv2d(response_inplanes[0], planes, kernel_size=1, stride=1, bias=False)
        self.g0 = nn.Conv2d(response_inplanes[0], planes, kernel_size=1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(perspective_inplanes)

        if self.use_scale:
            cprint("=> WARN: SpatialNL block uses 'SCALE' before softmax", 'yellow')
    def aggregate(self, response, perspective,m,save_attention=False):
        t = self.t(perspective)
        b, c, h, w = t.size()
        t = t.view(b, c, -1).permute(0, 2, 1)
        if m==4:
            p_f = self.p4
            g_f = self.g4
            bn_f = self.bn4
        if m==3:
            p_f = self.p3
            g_f = self.g3
            bn_f = self.bn3
        if m==2:
            p_f = self.p2
            g_f = self.g2
            bn_f = self.bn2
        if m==1:
            p_f = self.p1
            g_f = self.g1
            bn_f = self.bn1
        if m==0:
            p_f = self.p0
            g_f = self.g0
            bn_f = self.bn0
        p = p_f(response)
        g = g_f(response)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)
        att = torch.bmm(t, p)
        if self.use_scale:
            att = att.div(c**0.5)
        att = self.softmax(att)
        if save_attention==True:
            cur_perspective_att = att[0]
            for j in range(cur_perspective_att.size()[0]):
                image = cur_perspective_att[j].cpu().clone()
                image = image.view(int(math.sqrt(image.size()[0])),int(math.sqrt(image.size()[0])))
                image = image.detach().numpy()
                plt.imsave("/input/CNLNet/attention/perspective"+str(j)+"layer"+str(m)+".png",image, cmap="coolwarm")
        output = torch.bmm(att, g)
        output = output.contiguous()
        output = output.view(b, c, h, w)
        output = bn_f(self.z(output))

        return output
        
    def forward(self, responses,perspective, save_attention = False):
        #residual = perspective
        output4  = self.aggregate(responses[4], perspective,4,save_attention)
        output3  = self.aggregate(responses[3], perspective,3,save_attention)
        output2  = self.aggregate(responses[2], perspective,2,save_attention)
        output1  = self.aggregate(responses[1], perspective,1,save_attention)
        output0  = self.aggregate(responses[0], perspective,0,save_attention)
        output = output0+output1+output2+output3+output4
        
        return output

class CNLNet(nn.Module):

    def __init__(self, arch, block, layers, num_classes=1000,
                 response_layers=None, pool_size=7):
        self.inplanes = 64
        super(CNLNet, self).__init__()
        
        self.response_layers = response_layers
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.arch = arch
        
        layer1_inplanes = 256
        layer2_inplanes = 512
        layer3_inplanes = 1024
        layer4_inplanes = 2048
        self.output_planes = int(layer4_inplanes/8)
        self.CNL_5 = CNL_5([layer2_inplanes,layer2_inplanes,layer3_inplanes, layer3_inplanes,layer3_inplanes,],\
                           perspective_inplanes=layer4_inplanes,\
                        planes = self.output_planes,\
                        output_planes=layer4_inplanes,use_scale=True)
        
        
        self.avgpool = nn.AvgPool2d(pool_size, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc_m = nn.Linear(512 * block.expansion, num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if True:
            for name, m in self._modules['CNL_5'].named_modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                
    
    def _res3__of_res50(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        layers.append(block(self.inplanes, planes))

        layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _res4__of_res50(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        layers.append(block(self.inplanes, planes))
        
        layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes))

        layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input,save_attention=False):
        if save_attention==True:
            cur_perspective_img = input[0]
            image = cur_perspective_img.cpu().clone().detach().numpy()
            image = image.transpose((1,2,0))
            print(np.shape(image))
            plt.imsave("/input/CNLNet/attention/perspective_img.png",image, cmap="coolwarm")
        f1 = self.conv1(input)
        f2 = self.bn1(f1)
        f3 = self.relu(f2)
        f4 = self.maxpool(f3)

        f5 = self.layer1(f4)
        f6_1 = self._modules['layer2'][0](f5)
        f6_2 = self._modules['layer2'][1](f6_1)
        f6_3 = self._modules['layer2'][2](f6_2)
        f6_4 = self._modules['layer2'][3](f6_3)
        
        #Res50
        if self.arch == '50':
            f7_1 = self._modules['layer3'][0](f6_4)
            f7_2 = self._modules['layer3'][1](f7_1)
            f7_3 = self._modules['layer3'][2](f7_2)
            f7_4 = self._modules['layer3'][3](f7_3)
            f7_5 = self._modules['layer3'][4](f7_4)
            f7_6 = self._modules['layer3'][5](f7_5)
            
            f8_1 = self._modules['layer4'][0](f7_6)
            f8_2 = self._modules['layer4'][1](f8_1)
            f8_2 = f8_2+self.CNL_5([f6_2,f6_4,f7_2,f7_4,f7_6],f8_2,save_attention)
    
        #Res101
        if self.arch == '101':
            f7_1 = self._modules['layer3'][0](f6_4)
            f7_2 = self._modules['layer3'][1](f7_1)
            f7_3 = self._modules['layer3'][2](f7_2)
            f7_4 = self._modules['layer3'][3](f7_3)
            f7_5 = self._modules['layer3'][4](f7_4)
            f7_6 = self._modules['layer3'][5](f7_5)
            f7_7 = self._modules['layer3'][6](f7_6)
            f7_8 = self._modules['layer3'][7](f7_7)
            f7_9 = self._modules['layer3'][8](f7_8)
            f7_10 = self._modules['layer3'][9](f7_9)
            f7_11 = self._modules['layer3'][10](f7_10)
            f7_12 = self._modules['layer3'][11](f7_11)
            f7_13 = self._modules['layer3'][12](f7_12)
            f7_14 = self._modules['layer3'][13](f7_13)
            f7_15 = self._modules['layer3'][14](f7_14)
            f7_16 = self._modules['layer3'][15](f7_15)
            f7_17 = self._modules['layer3'][16](f7_16)
            f7_18 = self._modules['layer3'][17](f7_17)
            f7_19 = self._modules['layer3'][18](f7_18)
            f7_20 = self._modules['layer3'][19](f7_19)
            f7_21 = self._modules['layer3'][20](f7_20)
            f7_22 = self._modules['layer3'][21](f7_21)
            f7_23 = self._modules['layer3'][22](f7_22)
            
            f8_1 = self._modules['layer4'][0](f7_23)
            f8_2 = self._modules['layer4'][1](f8_1)
            f8_2 = f8_2+self.CNL_5([f6_2,f6_4,f7_6,f7_13,f7_20],f8_2,save_attention)
            
        f8_3 = self._modules['layer4'][2](f8_2)
        f10 = self.avgpool(f8_3)
        f10 = f10.view(f10.size(0), -1)
        f11 = self.dropout(f10)
        f12 = self.fc_m(f11)
        return f12

def load_partial_weight(model, pretrained, nl_layer_id):
    """Loads the partial weights for CNL network.
    """
    _pretrained = pretrained
    _model_dict = model.state_dict()
    _pretrained_dict = OrderedDict()
    for k, v in _pretrained.items():
        print(k)
        ks = k.split('.')
        layer_name = '.'.join(ks[0:2])
        _pretrained_dict[k] = v
    _model_dict.update(_pretrained_dict)
    return _model_dict

def cnlnet50(arch, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = CNLNet(arch,Bottleneck, [3, 4, 6, 3],**kwargs)
    if pretrained:
        _pretrained = torch.load('./pretrained/resnet50-19c8e357.pth')
        _model_dict = load_partial_weight(model, _pretrained, 5)
        model.load_state_dict(_model_dict,strict=False)
    return model


def cnlnet101(arch, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = CNLNet(arch,Bottleneck, [3, 4, 23, 3],**kwargs)
    if pretrained:
        _pretrained = torch.load('./pretrained/resnet101-5d3b4d8f.pth')
        _model_dict = load_partial_weight(model, _pretrained, 22)
        model.load_state_dict(_model_dict)
    return model
