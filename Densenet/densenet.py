import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import os

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else 'cpu')

Iteration = 300000
BATCH_SIZE = 64
momentum = 0.9
lr = 0.1

class Densenet(nn.Module):
    def __init__(self, image_size = 128):
        super(Densenet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.maxpool(x)
        return x

class Denseblock1(nn.Module):
    def __init__(self, image_size = 32):
        super(Denseblock1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(224, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self,x):
        concat1 = x
        x = self.conv1(x)
        x = torch.concat((concat1, x), dim=1)
        concat2 = x
        x = self.conv2(x)
        x = torch.concat((concat2, x), dim=1)
        concat3 = x
        x = self.conv3(x)
        x = torch.concat((concat3, x), dim=1)
        concat4 = x
        x = self.conv4(x)
        x = torch.concat((concat4, x), dim=1)
        concat5 = x
        x = self.conv5(x)
        x = torch.concat((concat5, x), dim=1)
        concat6 = x
        x = self.conv6(x)
        x = torch.concat((concat6, x), dim=1)
        x = self.layer3(x)
        x = self.avgpool(x)
        return x
class Denseblock2(nn.Module):
    def __init__(self, image_size=16):
        super(Denseblock2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(224, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(288, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(352, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(416, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(448, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(480, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        concat1 = x
        x = self.conv1(x)
        x = torch.concat((concat1, x), dim=1)
        concat2 = x
        x = self.conv2(x)
        x = torch.concat((concat2, x), dim=1)
        concat3 = x
        x = self.conv3(x)
        x = torch.concat((concat3, x), dim=1)
        concat4 = x
        x = self.conv4(x)
        x = torch.concat((concat4, x), dim=1)
        concat5 = x
        x = self.conv5(x)
        x = torch.concat((concat5, x), dim=1)
        concat6 = x
        x = self.conv6(x)
        x = torch.concat((concat6, x), dim=1)
        concat7 = x
        x = self.conv7(x)
        x = torch.concat((concat7, x), dim=1)
        concat8 = x
        x = self.conv8(x)
        x = torch.concat((concat8, x), dim=1)
        concat9 = x
        x = self.conv9(x)
        x = torch.concat((concat9, x), dim=1)
        concat10 = x
        x = self.conv10(x)
        x = torch.concat((concat10, x), dim=1)
        concat11 = x
        x = self.conv11(x)
        x = torch.concat((concat11, x), dim=1)
        concat12 = x
        x = self.conv12(x)
        x = torch.concat((concat12, x), dim=1)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x
class Denseblock3(nn.Module):
    def __init__(self, image_size=8):
        super(Denseblock3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(288, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(352, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(416, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(448, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(480, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(544, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(608, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(672, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(704, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(736, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(800, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv19 = nn.Sequential(
            nn.Conv2d(832, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv20 = nn.Sequential(
            nn.Conv2d(864, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv21 = nn.Sequential(
            nn.Conv2d(896, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(928, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(960, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv24 = nn.Sequential(
            nn.Conv2d(992, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.conv0 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        # x = self.conv0(x)
        concat1 = x
        x = self.conv1(x)
        x = torch.concat((concat1, x), dim=1)
        concat2 = x
        x = self.conv2(x)
        x = torch.concat((concat2, x), dim=1)
        concat3 = x
        x = self.conv3(x)
        x = torch.concat((concat3, x), dim=1)
        concat4 = x
        x = self.conv4(x)
        x = torch.concat((concat4, x), dim=1)
        concat5 = x
        x = self.conv5(x)
        x = torch.concat((concat5, x), dim=1)
        concat6 = x
        x = self.conv6(x)
        x = torch.concat((concat6, x), dim=1)
        concat7 = x
        x = self.conv7(x)
        x = torch.concat((concat7, x), dim=1)
        concat8 = x
        x = self.conv8(x)
        x = torch.concat((concat8, x), dim=1)
        concat9 = x
        x = self.conv9(x)
        x = torch.concat((concat9, x), dim=1)
        concat10 = x
        x = self.conv10(x)
        x = torch.concat((concat10, x), dim=1)
        concat11 = x
        x = self.conv11(x)
        x = torch.concat((concat11, x), dim=1)
        concat12 = x
        x = self.conv12(x)
        x = torch.concat((concat12, x), dim=1)
        concat13 = x
        x = self.conv13(x)
        x = torch.concat((concat13, x), dim=1)
        concat14 = x
        x = self.conv14(x)
        x = torch.concat((concat14, x), dim=1)
        concat15 = x
        x = self.conv15(x)
        x = torch.concat((concat15, x), dim=1)
        concat16 = x
        x = self.conv16(x)
        x = torch.concat((concat16, x), dim=1)
        concat17 = x
        x = self.conv17(x)
        x = torch.concat((concat17, x), dim=1)
        concat18 = x
        x = self.conv18(x)
        x = torch.concat((concat18, x), dim=1)
        concat19 = x
        x = self.conv19(x)
        x = torch.concat((concat19, x), dim=1)
        concat20 = x
        x = self.conv20(x)
        x = torch.concat((concat20, x), dim=1)
        concat21 = x
        x = self.conv21(x)
        x = torch.concat((concat21, x), dim=1)
        concat22 = x
        x = self.conv22(x)
        x = torch.concat((concat22, x), dim=1)
        concat23 = x
        x = self.conv23(x)
        x = torch.concat((concat23, x), dim=1)
        concat24 = x
        x = self.conv24(x)
        x = torch.concat((concat24, x), dim=1)
        x = self.layer5(x)
        x = self.avgpool(x)
        return x
class Denseblock4(nn.Module):
    def __init__(self, image_size=4):
        super(Denseblock4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(544, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(608, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(672, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(704, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(736, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(800, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(832, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(864, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(896, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(928, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(960, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(992, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        # self.conv0 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.global_avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.FC = nn.Linear(1024,200)

    def forward(self, x):
        # x = self.conv0(x)
        concat1 = x
        x = self.conv1(x)
        x = torch.concat((concat1, x), dim=1)
        concat2 = x
        x = self.conv2(x)
        x = torch.concat((concat2, x), dim=1)
        concat3 = x
        x = self.conv3(x)
        x = torch.concat((concat3, x), dim=1)
        concat4 = x
        x = self.conv4(x)
        x = torch.concat((concat4, x), dim=1)
        concat5 = x
        x = self.conv5(x)
        x = torch.concat((concat5, x), dim=1)
        concat6 = x
        x = self.conv6(x)
        x = torch.concat((concat6, x), dim=1)
        concat7 = x
        x = self.conv7(x)
        x = torch.concat((concat7, x), dim=1)
        concat8 = x
        x = self.conv8(x)
        x = torch.concat((concat8, x), dim=1)
        concat9 = x
        x = self.conv9(x)
        x = torch.concat((concat9, x), dim=1)
        concat10 = x
        x = self.conv10(x)
        x = torch.concat((concat10, x), dim=1)
        concat11 = x
        x = self.conv11(x)
        x = torch.concat((concat11, x), dim=1)
        concat12 = x
        x = self.conv12(x)
        x = torch.concat((concat12, x), dim=1)
        concat13 = x
        x = self.conv13(x)
        x = torch.concat((concat13, x), dim=1)
        concat14 = x
        x = self.conv14(x)
        x = torch.concat((concat14, x), dim=1)
        concat15 = x
        x = self.conv15(x)
        x = torch.concat((concat15, x), dim=1)
        concat16 = x
        x = self.conv16(x)
        x = torch.concat((concat16, x), dim=1)
        x = self.global_avgpool(x)
        x = torch.flatten(x,1)
        x = self.FC(x)
        return x

train_path = r"C:\Users\AIVS\Desktop\tinyImageNet\train/"
train_list = os.listdir(train_path)
test_path = r"C:\Users\AIVS\Desktop\tinyImageNet\test/"
test_list = os.listdir(test_path)
train_gt_path = r"C:\Users\AIVS\Desktop\tinyImageNet\train_gt.txt"
f = open(train_gt_path, 'r')
train_gt = f.readlines()
test_gt_path = r"C:\Users\AIVS\Desktop\tinyImageNet\test_gt.txt"
f_gt = open(test_gt_path, 'r')
test_gt = f_gt.readlines()

model = Densenet().to(DEVICE).requires_grad_(True)
denseblock1 = Denseblock1().to(DEVICE).requires_grad_(True)
denseblock2 = Denseblock2().to(DEVICE).requires_grad_(True)
denseblock3 = Denseblock3().to(DEVICE).requires_grad_(True)
denseblock4 = Denseblock4().to(DEVICE).requires_grad_(True)
optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

from itertools import chain
all_modules = (model,denseblock1,denseblock2,denseblock3,denseblock4)
all_parameters = chain.from_iterable(m.parameters()for m in all_modules)

for i in range(Iteration):
    if i == 100000: optimizer.param_groups[0]['lr'] = 0.005
    if i == 200000: optimizer.param_groups[0]['lr'] = 0.001
    if i >= 250000:
        optimizer.param_groups[0]['lr'] = 0.001 - (i / Iteration) * 0.001
    model.train()
    denseblock1.train()
    denseblock2.train()
    denseblock3.train()
    denseblock4.train()
    train_random_idx = np.random.randint(0, len(train_list), BATCH_SIZE)
    batch_img = np.zeros((BATCH_SIZE, 3, 128, 128), dtype=np.float32)
    batch_gt = np.zeros(BATCH_SIZE, dtype=np.int64)

    for j in range(BATCH_SIZE):
        idx = train_random_idx[j]
        img_path = os.path.join(train_path, train_list[idx])
        img = cv2.imread(img_path)
        img_tmp = (img.astype(np.float32) / 255.0) * 2 - 1
        if np.random.randint(2):
            img_tmp = cv2.flip(img_tmp, 1)
        if np.random.randint(2):
            blank_img = np.zeros((224, 224, 3), dtype=np.float32)
            x0, y0 = (224 - 128) // 2, (224 - 128) // 2
            blank_img[y0:y0 + 128, x0:x0 + 128] = img_tmp
            sy = np.random.randint(0, 224 - 128 + 1)
            sx = np.random.randint(0, 224 - 128 + 1)
            img_tmp = blank_img[sy:sy + 128, sx:sx + 128]

        if np.random.randint(2):
            img_tmp = cv2.rotate(img_tmp, cv2.ROTATE_90_CLOCKWISE)

        img_tmp = np.transpose(img_tmp, (2, 0, 1))
        batch_img[j] = img_tmp
        batch_gt[j] = int(train_gt[idx]) - 1

    data = torch.from_numpy(batch_img).to(DEVICE)
    target = torch.from_numpy(batch_gt).to(DEVICE)
    optimizer.zero_grad()
    output_from_densenet = model(data)
    output1 = denseblock1(output_from_densenet)
    output2 = denseblock2(output1)
    output3 = denseblock3(output2)
    output4 = denseblock4(output3)
    loss_val = criterion(output4, target)
    loss_val.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(f'Iter: {i}, Loss: {loss_val.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    if i % 10000 == 0 and i != 0:
        model.eval()
        denseblock1.eval()
        denseblock2.eval()
        denseblock3.eval()
        denseblock4.eval()
        count = 0
        with torch.no_grad():
            for j in range(len(test_list)):
                test_img_path = os.path.join(test_path, test_list[j])
                test_img = cv2.imread(test_img_path)
                if test_img is None: continue
                test_img_tmp = (test_img.astype(np.float32) / 255.0) * 2 - 1
                test_img_tmp = np.transpose(test_img_tmp, (2, 0, 1))
                test_img_tmp = torch.from_numpy(test_img_tmp).unsqueeze(0).to(DEVICE)
                test_output_densenet = model(test_img_tmp)
                test_output_1 = denseblock1(test_output_densenet)
                test_output_2 = denseblock2(test_output_1)
                test_output_3 = denseblock3(test_output_2)
                test_output_4 = denseblock4(test_output_3)
                pre = torch.argmax(test_output_4, -1)
                if pre.item() == int(test_gt[j]) - 1:
                    count += 1
        accuracy = count / len(test_list) * 100
        print(f"Iter: {i}, Count: {count}, Accuracy: {accuracy:.2f}%")
        torch.save(model.state_dict(), f"./model_save/Iter_{i}_acc_{accuracy:.1f}.pth")
        model.train()


