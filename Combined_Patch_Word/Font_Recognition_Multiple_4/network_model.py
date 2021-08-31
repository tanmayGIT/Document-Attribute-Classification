from __future__ import print_function
from __future__ import division
import torch.nn as nn
import numpy as np

from vgg_blocks import *
import torch
from torchvision import models
import torch.nn.functional as Fun


class VariousModels:
    def __init__(self, model_name, num_classes, feature_extract):
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet_multi_task":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features

            # model_ft.fc = nn.Linear(num_ftrs, 512)
            input_size = 224

        elif model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

class NetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=Fun.relu):
        super(NetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        return out


class CombineMultiOutputModel(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModel, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.x2 = nn.Linear(256, 256)  # 512; no because it is not for second input. It is to apply 2nd linear layer
        nn.init.xavier_normal_(self.x2.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.bn2 = nn.BatchNorm1d(256, eps=2e-1)

        self.x11 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x11.weight)

        self.x12 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x12.weight)

        self.x13 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x13.weight)

        self.x14 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x14.weight)

        self.x21 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x21.weight)

        self.x22 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x22.weight)

        self.x23 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x23.weight)

        self.x24 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x24.weight)

        # this is to apply linear layer after combination two vectors
        self.x1c = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x1c.weight)

        self.x2c = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2c.weight)

        self.x3c = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x3c.weight)

        self.x4c = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x4c.weight)

        # heads
        self.y1o = nn.Linear(256, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(256, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(256, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(256, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(0.5)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)
        x1_out = self.bn1(Fun.relu(self.x1(x1_out)))
        x1_out = self.bn2(Fun.relu(self.x2(x1_out)))

        x2_out = self.resnet_model(x2_input)
        x2_out = self.bn1(Fun.relu(self.x1(x2_out)))
        x2_out = self.bn2(Fun.relu(self.x2(x2_out)))

        x11 = self.bn1(Fun.relu(self.x11(x1_out)))
        x12 = self.bn1(Fun.relu(self.x12(x1_out)))
        x13 = self.bn1(Fun.relu(self.x13(x1_out)))
        x14 = self.bn1(Fun.relu(self.x14(x1_out)))

        x21 = self.bn1(Fun.relu(self.x21(x2_out)))
        x22 = self.bn1(Fun.relu(self.x22(x2_out)))
        x23 = self.bn1(Fun.relu(self.x23(x2_out)))
        x24 = self.bn1(Fun.relu(self.x24(x2_out)))

        x11 = Fun.relu((x11 + x21)/2)
        # print('The value of x11', x11)

        x12 = Fun.relu((x12 + x22)/2)
        # print('The value of x12', x12)

        x13 = Fun.relu((x13 + x23)/2)
        # print('The value of x13', x13)

        x14 = Fun.relu((x14 + x24)/2)
        # print('The value of x14', x14)

        x1c = self.d_out(self.bn1(Fun.relu(self.x1c(x11))))
        x2c = self.d_out(self.bn1(Fun.relu(self.x2c(x12))))
        x3c = self.d_out(self.bn1(Fun.relu(self.x3c(x13))))
        x4c = self.d_out(self.bn1(Fun.relu(self.x4c(x14))))

        # heads
        y1o = Fun.softmax(self.y1o(x1c), dim=1)
        y2o = Fun.softmax(self.y2o(x2c), dim=1)
        y3o = Fun.softmax(self.y3o(x3c), dim=1)
        y4o = Fun.softmax(self.y4o(x4c), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, kernel_size=1, stride=1, padding=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride, padding=1),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ConvRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, kernel_size=1, stride=1, padding=1):
        super(ConvRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class FlatAvgPool(nn.Module):

    def __init__(self):
        super(FlatAvgPool, self).__init__()

        self.flat_avg_pool = nn.AdaptiveAvgPool1d(6)
        self.just_flatten = nn.Flatten()

    def forward(self, x):
        x_out = self.flat_avg_pool(x)
        x_out = self.just_flatten(x_out)

        return x_out


class FeatureLinearLayer(nn.Module):

    def __init__(self):
        super(FeatureLinearLayer, self).__init__()

        self.classifier = nn.Sequential(

            # FC 1
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6, 768),
            nn.ReLU(inplace=True),

            # FC 2
            nn.Dropout(p=0.5),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),

            # FC 3
            nn.Linear(384, 192),
        )

    def forward(self, x):
        return self.classifier(x)


class CombineMultiOutputModelConvAlexNet(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModelConvAlexNet, self).__init__()

        self.resnet_model = model_core

        self.Conv_1 = ConvBNRelu(2, 64, 11, 4, 2)
        self.Conv_2 = ConvBNRelu(64, 192, 5, 1, 2)
        self.Conv_3 = ConvBNRelu(192, 384, 3, 1, 1)
        self.Conv_4 = ConvBNRelu(384, 256, 3, 1, 1)
        self.Conv_5 = ConvBNRelu(256, 256, 3, 1, 1)

        self.AvgPoolFlat = FlatAvgPool()
        self.Last_Linear_Layer = FeatureLinearLayer()

        # heads
        self.y1o = nn.Linear(192, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(192, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(192, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(192, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(0.5)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)  # 2048 features
        x2_out = self.resnet_model(x2_input)  # 2048 features

        # generate 2 channels from these two 1D vectors
        n_batch_size = x1_out.shape[0]
        n_rows = x1_out.shape[1]

        combine_features = np.empty((n_batch_size, 2, n_rows))  # n_batch, nChannel, img_rows, img_cols
        combine_features = torch.from_numpy(combine_features).float()
        combine_features = torch.tensor(combine_features)
        # making 2D vector into 3D vector
        for iBatch in range(0, n_batch_size):
            combine_features[iBatch, 0, :] = x1_out[iBatch, :]  # making 1st channel
            combine_features[iBatch, 1, :] = x2_out[iBatch, :]  # making the 2nd channel

        combine_features = combine_features.cuda()
        x_conv_out = self.Conv_1(combine_features)
        x_conv_out = self.Conv_2(x_conv_out)
        x_conv_out = self.Conv_3(x_conv_out)
        x_conv_out = self.Conv_4(x_conv_out)
        x_conv_out = self.Conv_5(x_conv_out)

        linear_features = self.AvgPoolFlat(x_conv_out)
        linear_features = self.Last_Linear_Layer(linear_features)

        # heads
        y1o = Fun.softmax(self.y1o(linear_features), dim=1)
        y2o = Fun.softmax(self.y2o(linear_features), dim=1)
        y3o = Fun.softmax(self.y3o(linear_features), dim=1)
        y4o = Fun.softmax(self.y4o(linear_features), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o


class CombineMultiOutputModelConvVggNetSimple(nn.Module):

    def __init__(self, model_core):

        super(CombineMultiOutputModelConvVggNetSimple, self).__init__()

        self.resnet_model = model_core

        self.double_conv_1 = DoubleConv(2, 64)
        self.double_conv_2 = DoubleConv(64, 128)

        self.fourth_conv_1 = FourthConv(128, 256)
        self.fourth_conv_2 = FourthConv(256, 512)
        self.fourth_conv_3 = FourthConv(512, 512)

        self.fl_1 = FullyConnectedLayer(25088, 4096, first_l=True, final_l=False)
        self.fl_2 = FullyConnectedLayer(4096, 4096, first_l=False, final_l=False)
        self.fl_3 = FullyConnectedLayer(4096, 10, first_l=False, final_l=True)

        self.AvgPoolFlat = VggFlatAvgPool()
        self.Last_Linear_Layer = VggFeatureLinearLayer()

        # heads
        self.y1o = nn.Linear(192, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(192, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(192, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(192, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(0.5)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)  # 2048 features
        x2_out = self.resnet_model(x2_input)  # 2048 features

        # generate 2 channels from these two 1D vectors
        n_batch_size = x1_out.shape[0]
        n_rows = x1_out.shape[1]

        combine_features = np.empty((n_batch_size, 2, n_rows))  # n_batch, nChannel, img_rows, img_cols
        combine_features = torch.from_numpy(combine_features).float()
        combine_features = torch.tensor(combine_features)
        # making 2D vector into 3D vector
        for iBatch in range(0, n_batch_size):
            combine_features[iBatch, 0, :] = x1_out[iBatch, :]  # making 1st channel
            combine_features[iBatch, 1, :] = x2_out[iBatch, :]  # making the 2nd channel

        combine_features = combine_features.cuda()
        x1 = self.double_conv_1(combine_features)
        x2 = self.double_conv_2(x1)

        x3 = self.fourth_conv_1(x2)
        x4 = self.fourth_conv_2(x3)
        x_conv_out = self.fourth_conv_3(x4)

        # input to linear function is x5.size([1, 512, 7, 7]) -> 512*7*7=25088

        # x6 = self.fl_1(x5)
        # x7 = self.fl_2(x6)
        # x_out = self.fl_3(x7)

        linear_features = self.AvgPoolFlat(x_conv_out)
        linear_features = self.Last_Linear_Layer(linear_features)

        # heads
        y1o = Fun.softmax(self.y1o(linear_features), dim=1)
        y2o = Fun.softmax(self.y2o(linear_features), dim=1)
        y3o = Fun.softmax(self.y3o(linear_features), dim=1)
        y4o = Fun.softmax(self.y4o(linear_features), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o

class CombineMultiOutputModelEarlyConcat_1(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModelEarlyConcat_1, self).__init__()

        self.resnet_model = model_core

        self.x1Early1 = nn.Linear(4096, 2048)
        nn.init.xavier_normal_(self.x1Early1.weight)
        self.bn6 = nn.BatchNorm1d(2048, eps=2e-1)

        self.x1Early2 = nn.Linear(2048, 1024)
        nn.init.xavier_normal_(self.x1Early2.weight)
        self.bn7 = nn.BatchNorm1d(1024, eps=2e-1)

        self.x1Early3 = nn.Linear(1024, 512)
        nn.init.xavier_normal_(self.x1Early3.weight)
        self.bn8 = nn.BatchNorm1d(512, eps=2e-1)

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.x2 = nn.Linear(256, 256)  # 512; no because it is not for second input. It is to apply 2nd linear layer
        nn.init.xavier_normal_(self.x2.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.bn2 = nn.BatchNorm1d(128, eps=2e-1)
        self.bn3 = nn.BatchNorm1d(64, eps=2e-1)
        self.bn4 = nn.BatchNorm1d(32, eps=2e-1)
        self.bn5 = nn.BatchNorm1d(16, eps=2e-1)

        self.x11 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x11.weight)

        self.brk1 = nn.Linear(256, 128)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk1.weight)

        self.brk2 = nn.Linear(128, 64)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk2.weight)

        self.brk3 = nn.Linear(64, 32)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk3.weight)

        self.brk4 = nn.Linear(32, 16)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk4.weight)

        # heads
        self.y1o = nn.Linear(16, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(16, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(16, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(16, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(0.5)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)  # 2048 features
        x2_out = self.resnet_model(x2_input)  # 2048 features
        x_concat = torch.cat((x1_out, x2_out), 1)  # 4096 features

        x1_out = self.bn6(Fun.relu(self.x1Early1(x_concat)))  # 4096 to 2048
        x1_out = self.bn7(Fun.relu(self.x1Early2(x1_out)))  # 2048 to 1024
        x1_out = self.bn8(Fun.relu(self.x1Early3(x1_out)))  # 1024 to 512
        x1_out = self.bn1(Fun.relu(self.x1(x1_out)))  # 512 to 256

        x11 = (self.bn1(Fun.relu(self.x11(x1_out))))  # 256 to 256
        x12 = (self.bn1(Fun.relu(self.x11(x1_out))))  # 256 to 256
        x13 = (self.bn1(Fun.relu(self.x11(x1_out))))  # 256 to 256
        x14 = (self.bn1(Fun.relu(self.x11(x1_out))))  # 256 to 256

        x1d1 = (self.bn2(Fun.relu(self.brk1(x11))))  # 256 -> 128
        x2d1 = (self.bn2(Fun.relu(self.brk1(x12))))  # 256 -> 128
        x3d1 = (self.bn2(Fun.relu(self.brk1(x13))))  # 256 -> 128
        x4d1 = (self.bn2(Fun.relu(self.brk1(x14))))  # 256 -> 128

        x1d2 = (self.bn3(Fun.relu(self.brk2(x1d1))))  # 128 -> 64
        x2d2 = (self.bn3(Fun.relu(self.brk2(x2d1))))  # 128 -> 64
        x3d2 = (self.bn3(Fun.relu(self.brk2(x3d1))))  # 128 -> 64
        x4d2 = (self.bn3(Fun.relu(self.brk2(x4d1))))  # 128 -> 64

        x1d3 = (self.bn4(Fun.relu(self.brk3(x1d2))))  # 64 -> 32
        x2d3 = (self.bn4(Fun.relu(self.brk3(x2d2))))  # 64 -> 32
        x3d3 = (self.bn4(Fun.relu(self.brk3(x3d2))))  # 64 -> 32
        x4d3 = (self.bn4(Fun.relu(self.brk3(x4d2))))  # 64 -> 32

        x1d4 = (self.bn5(Fun.relu(self.brk4(x1d3))))  # 32 -> 16
        x2d4 = (self.bn5(Fun.relu(self.brk4(x2d3))))  # 32 -> 16
        x3d4 = (self.bn5(Fun.relu(self.brk4(x3d3))))  # 32 -> 16
        x4d4 = (self.bn5(Fun.relu(self.brk4(x4d3))))  # 32 -> 16

        # heads
        y1o = Fun.softmax(self.y1o(x1d4), dim=1)
        y2o = Fun.softmax(self.y2o(x2d4), dim=1)
        y3o = Fun.softmax(self.y3o(x3d4), dim=1)
        y4o = Fun.softmax(self.y4o(x4d4), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o


class CombineMultiOutputModelEarlyConcat_2(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModelEarlyConcat_2, self).__init__()

        self.resnet_model = model_core

        self.x1Early1 = nn.Linear(4096, 2048)
        nn.init.xavier_normal_(self.x1Early1.weight)
        self.bn6 = nn.BatchNorm1d(2048, eps=2e-1)

        # heads
        self.y1o = nn.Linear(2048, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(2048, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(2048, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(2048, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(0.5)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)  # 2048 features
        x2_out = self.resnet_model(x2_input)  # 2048 features
        x_concat = torch.cat((x1_out, x2_out), 1)  # 4096 features

        x1_out = self.bn6(Fun.relu(self.x1Early1(x_concat)))  # 4096 to 2048

        # heads
        y1o = Fun.softmax(self.y1o(x1_out), dim=1)
        y2o = Fun.softmax(self.y2o(x1_out), dim=1)
        y3o = Fun.softmax(self.y3o(x1_out), dim=1)
        y4o = Fun.softmax(self.y4o(x1_out), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o


class CombineMultiOutputModelEarlyConcat_3(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModelEarlyConcat_3, self).__init__()

        self.resnet_model = model_core

        self.x1Early1 = nn.Linear(4096, 2048)
        nn.init.xavier_normal_(self.x1Early1.weight)
        self.bn6 = nn.BatchNorm1d(2048, eps=2e-1)

        self.x1Early2 = nn.Linear(2048, 1024)
        nn.init.xavier_normal_(self.x1Early2.weight)
        self.bn7 = nn.BatchNorm1d(1024, eps=2e-1)

        # heads
        self.y1o = nn.Linear(1024, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(1024, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(1024, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(1024, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(0.5)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)  # 2048 features
        x2_out = self.resnet_model(x2_input)  # 2048 features
        x_concat = torch.cat((x1_out, x2_out), 1)  # 4096 features

        x1_out = self.bn6(Fun.relu(self.x1Early1(x_concat)))  # 4096 to 2048
        x1_out = self.bn7(Fun.relu(self.x1Early2(x1_out)))  # 2048 to 1024

        # heads
        y1o = Fun.softmax(self.y1o(x1_out), dim=1)
        y2o = Fun.softmax(self.y2o(x1_out), dim=1)
        y3o = Fun.softmax(self.y3o(x1_out), dim=1)
        y4o = Fun.softmax(self.y4o(x1_out), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o


class CombineMultiOutputModelConcat(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModelConcat, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.x2 = nn.Linear(256, 256)  # 512; no because it is not for second input. It is to apply 2nd linear layer
        nn.init.xavier_normal_(self.x2.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.bn2 = nn.BatchNorm1d(128, eps=2e-1)
        self.bn3 = nn.BatchNorm1d(64, eps=2e-1)
        self.bn4 = nn.BatchNorm1d(32, eps=2e-1)
        self.bn5 = nn.BatchNorm1d(16, eps=2e-1)

        self.x11 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x11.weight)

        # this is to apply linear layer after combination two vectors
        self.x1c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1c.weight)

        self.brk1 = nn.Linear(256, 128)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk1.weight)

        self.brk2 = nn.Linear(128, 64)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk2.weight)

        self.brk3 = nn.Linear(64, 32)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk3.weight)

        self.brk4 = nn.Linear(32, 16)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk4.weight)

        # heads
        self.y1o = nn.Linear(16, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(16, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(16, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(16, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(0.3)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)
        x1_out = self.bn1(Fun.relu(self.x1(x1_out)))
        x1_out = self.bn1(Fun.relu(self.x2(x1_out)))

        x2_out = self.resnet_model(x2_input)
        x2_out = self.bn1(Fun.relu(self.x1(x2_out)))
        x2_out = self.bn1(Fun.relu(self.x2(x2_out)))

        x11 = (self.bn1(Fun.relu(self.x11(x1_out))))
        x12 = (self.bn1(Fun.relu(self.x11(x1_out))))
        x13 = (self.bn1(Fun.relu(self.x11(x1_out))))
        x14 = (self.bn1(Fun.relu(self.x11(x1_out))))

        x21 = (self.bn1(Fun.relu(self.x11(x2_out))))
        x22 = (self.bn1(Fun.relu(self.x11(x2_out))))
        x23 = (self.bn1(Fun.relu(self.x11(x2_out))))
        x24 = (self.bn1(Fun.relu(self.x11(x2_out))))

        x11 = torch.cat((x11, x21), 1)
        # print('The value of x11', x11)

        x12 = torch.cat((x12, x22), 1)
        # print('The value of x12', x12)

        x13 = torch.cat((x13, x23), 1)
        # print('The value of x13', x13)

        x14 = torch.cat((x14, x24), 1)
        # print('The value of x14', x14)

        x1c = (self.bn1(Fun.relu(self.x1c(x11))))  # 512 -> 256
        x2c = (self.bn1(Fun.relu(self.x1c(x12))))  # 512 -> 256
        x3c = (self.bn1(Fun.relu(self.x1c(x13))))  # 512 -> 256
        x4c = (self.bn1(Fun.relu(self.x1c(x14))))  # 512 -> 256

        x1d1 = (self.bn2(Fun.relu(self.brk1(x1c))))   # 256 -> 128
        x2d1 = (self.bn2(Fun.relu(self.brk1(x2c))))   # 256 -> 128
        x3d1 = (self.bn2(Fun.relu(self.brk1(x3c))))   # 256 -> 128
        x4d1 = (self.bn2(Fun.relu(self.brk1(x4c))))   # 256 -> 128

        x1d2 = (self.bn3(Fun.relu(self.brk2(x1d1))))  # 128 -> 64
        x2d2 = (self.bn3(Fun.relu(self.brk2(x2d1))))  # 128 -> 64
        x3d2 = (self.bn3(Fun.relu(self.brk2(x3d1))))  # 128 -> 64
        x4d2 = (self.bn3(Fun.relu(self.brk2(x4d1))))  # 128 -> 64

        x1d3 = (self.bn4(Fun.relu(self.brk3(x1d2))))  # 64 -> 32
        x2d3 = (self.bn4(Fun.relu(self.brk3(x2d2))))  # 64 -> 32
        x3d3 = (self.bn4(Fun.relu(self.brk3(x3d2))))  # 64 -> 32
        x4d3 = (self.bn4(Fun.relu(self.brk3(x4d2))))  # 64 -> 32

        x1d4 = (self.bn5(Fun.relu(self.brk4(x1d3))))  # 32 -> 16
        x2d4 = (self.bn5(Fun.relu(self.brk4(x2d3))))  # 32 -> 16
        x3d4 = (self.bn5(Fun.relu(self.brk4(x3d3))))  # 32 -> 16
        x4d4 = (self.bn5(Fun.relu(self.brk4(x4d3))))  # 32 -> 16

        # heads
        y1o = Fun.softmax(self.y1o(x1d4), dim=1)
        y2o = Fun.softmax(self.y2o(x2d4), dim=1)
        y3o = Fun.softmax(self.y3o(x3d4), dim=1)
        y4o = Fun.softmax(self.y4o(x4d4), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o


class CombineMultiOutputModelConcat_DropOut(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModelConcat_DropOut, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.x2 = nn.Linear(256, 256)  # 512; no because it is not for second input. It is to apply 2nd linear layer
        nn.init.xavier_normal_(self.x2.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.bn2 = nn.BatchNorm1d(128, eps=2e-1)
        self.bn3 = nn.BatchNorm1d(64, eps=2e-1)
        self.bn4 = nn.BatchNorm1d(32, eps=2e-1)
        self.bn5 = nn.BatchNorm1d(16, eps=2e-1)

        self.x11 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x11.weight)

        self.x12 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x12.weight)

        self.x13 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x13.weight)

        self.x14 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x14.weight)

        self.x21 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x21.weight)

        self.x22 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x22.weight)

        self.x23 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x23.weight)

        self.x24 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x24.weight)

        # this is to apply linear layer after combination two vectors
        self.x1c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1c.weight)

        self.x2c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x2c.weight)

        self.x3c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x3c.weight)

        self.x4c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x4c.weight)

        self.brk1 = nn.Linear(256, 128)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk1.weight)

        self.brk2 = nn.Linear(128, 64)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk2.weight)

        self.brk3 = nn.Linear(64, 32)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk3.weight)

        self.brk4 = nn.Linear(32, 16)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk4.weight)

        # heads
        self.y1o = nn.Linear(16, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(16, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(16, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(16, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(0.3)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)
        x1_out = self.bn1(Fun.relu(self.x1(x1_out)))
        x1_out = self.bn1(Fun.relu(self.x2(x1_out)))

        x2_out = self.resnet_model(x2_input)
        x2_out = self.bn1(Fun.relu(self.x1(x2_out)))
        x2_out = self.bn1(Fun.relu(self.x2(x2_out)))

        x11 = self.d_out(self.bn1(Fun.relu(self.x11(x1_out))))
        x12 = self.d_out(self.bn1(Fun.relu(self.x12(x1_out))))
        x13 = self.d_out(self.bn1(Fun.relu(self.x13(x1_out))))
        x14 = self.d_out(self.bn1(Fun.relu(self.x14(x1_out))))

        x21 = self.d_out(self.bn1(Fun.relu(self.x21(x2_out))))
        x22 = self.d_out(self.bn1(Fun.relu(self.x22(x2_out))))
        x23 = self.d_out(self.bn1(Fun.relu(self.x23(x2_out))))
        x24 = self.d_out(self.bn1(Fun.relu(self.x24(x2_out))))

        x11 = torch.cat((x11, x21), 1)
        # print('The value of x11', x11)

        x12 = torch.cat((x12, x22), 1)
        # print('The value of x12', x12)

        x13 = torch.cat((x13, x23), 1)
        # print('The value of x13', x13)

        x14 = torch.cat((x14, x24), 1)
        # print('The value of x14', x14)

        x1c = self.d_out(self.bn1(Fun.relu(self.x1c(x11))))  # 512 -> 256
        x2c = self.d_out(self.bn1(Fun.relu(self.x2c(x12))))  # 512 -> 256
        x3c = self.d_out(self.bn1(Fun.relu(self.x3c(x13))))  # 512 -> 256
        x4c = self.d_out(self.bn1(Fun.relu(self.x4c(x14))))  # 512 -> 256

        x1d1 = (self.bn2(Fun.relu(self.brk1(x1c))))   # 256 -> 128
        x2d1 = (self.bn2(Fun.relu(self.brk1(x2c))))   # 256 -> 128
        x3d1 = (self.bn2(Fun.relu(self.brk1(x3c))))   # 256 -> 128
        x4d1 = (self.bn2(Fun.relu(self.brk1(x4c))))   # 256 -> 128

        x1d2 = (self.bn3(Fun.relu(self.brk2(x1d1))))  # 128 -> 64
        x2d2 = (self.bn3(Fun.relu(self.brk2(x2d1))))  # 128 -> 64
        x3d2 = (self.bn3(Fun.relu(self.brk2(x3d1))))  # 128 -> 64
        x4d2 = (self.bn3(Fun.relu(self.brk2(x4d1))))  # 128 -> 64

        x1d3 = (self.bn4(Fun.relu(self.brk3(x1d2))))  # 64 -> 32
        x2d3 = (self.bn4(Fun.relu(self.brk3(x2d2))))  # 64 -> 32
        x3d3 = (self.bn4(Fun.relu(self.brk3(x3d2))))  # 64 -> 32
        x4d3 = (self.bn4(Fun.relu(self.brk3(x4d2))))  # 64 -> 32

        x1d4 = (self.bn5(Fun.relu(self.brk4(x1d3))))  # 32 -> 16
        x2d4 = (self.bn5(Fun.relu(self.brk4(x2d3))))  # 32 -> 16
        x3d4 = (self.bn5(Fun.relu(self.brk4(x3d3))))  # 32 -> 16
        x4d4 = (self.bn5(Fun.relu(self.brk4(x4d3))))  # 32 -> 16

        # heads
        y1o = Fun.softmax(self.y1o(x1d4), dim=1)
        y2o = Fun.softmax(self.y2o(x2d4), dim=1)
        y3o = Fun.softmax(self.y3o(x3d4), dim=1)
        y4o = Fun.softmax(self.y4o(x4d4), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o


class CombineMultiOutputModelConcat_NoBatchNorm(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModelConcat_NoBatchNorm, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.x2 = nn.Linear(256, 256)  # 512; no because it is not for second input. It is to apply 2nd linear layer
        nn.init.xavier_normal_(self.x2.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.bn2 = nn.BatchNorm1d(128, eps=2e-1)
        self.bn3 = nn.BatchNorm1d(64, eps=2e-1)
        self.bn4 = nn.BatchNorm1d(32, eps=2e-1)
        self.bn5 = nn.BatchNorm1d(16, eps=2e-1)

        self.x11 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x11.weight)

        self.x12 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x12.weight)

        self.x13 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x13.weight)

        self.x14 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x14.weight)

        self.x21 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x21.weight)

        self.x22 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x22.weight)

        self.x23 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x23.weight)

        self.x24 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x24.weight)

        # this is to apply linear layer after combination two vectors
        self.x1c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1c.weight)

        self.x2c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x2c.weight)

        self.x3c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x3c.weight)

        self.x4c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x4c.weight)

        self.brk1 = nn.Linear(256, 128)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk1.weight)

        self.brk2 = nn.Linear(128, 64)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk2.weight)

        self.brk3 = nn.Linear(64, 32)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk3.weight)

        self.brk4 = nn.Linear(32, 16)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.brk4.weight)

        # heads
        self.y1o = nn.Linear(16, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(16, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(16, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(16, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(0.3)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)
        x1_out = (Fun.relu(self.x1(x1_out)))
        x1_out = (Fun.relu(self.x2(x1_out)))

        x2_out = self.resnet_model(x2_input)
        x2_out = (Fun.relu(self.x1(x2_out)))
        x2_out = (Fun.relu(self.x2(x2_out)))

        x11 = ((Fun.relu(self.x11(x1_out))))
        x12 = ((Fun.relu(self.x12(x1_out))))
        x13 = ((Fun.relu(self.x13(x1_out))))
        x14 = ((Fun.relu(self.x14(x1_out))))

        x21 = ((Fun.relu(self.x21(x2_out))))
        x22 = ((Fun.relu(self.x22(x2_out))))
        x23 = ((Fun.relu(self.x23(x2_out))))
        x24 = ((Fun.relu(self.x24(x2_out))))

        x11 = torch.cat((x11, x21), 1)
        # print('The value of x11', x11)

        x12 = torch.cat((x12, x22), 1)
        # print('The value of x12', x12)

        x13 = torch.cat((x13, x23), 1)
        # print('The value of x13', x13)

        x14 = torch.cat((x14, x24), 1)
        # print('The value of x14', x14)

        x1c = ((Fun.relu(self.x1c(x11))))  # 512 -> 256
        x2c = ((Fun.relu(self.x2c(x12))))  # 512 -> 256
        x3c = ((Fun.relu(self.x3c(x13))))  # 512 -> 256
        x4c = ((Fun.relu(self.x4c(x14))))  # 512 -> 256

        x1d1 = ((Fun.relu(self.brk1(x1c))))   # 256 -> 128
        x2d1 = ((Fun.relu(self.brk1(x2c))))   # 256 -> 128
        x3d1 = ((Fun.relu(self.brk1(x3c))))   # 256 -> 128
        x4d1 = ((Fun.relu(self.brk1(x4c))))   # 256 -> 128

        x1d2 = ((Fun.relu(self.brk2(x1d1))))  # 128 -> 64
        x2d2 = ((Fun.relu(self.brk2(x2d1))))  # 128 -> 64
        x3d2 = ((Fun.relu(self.brk2(x3d1))))  # 128 -> 64
        x4d2 = ((Fun.relu(self.brk2(x4d1))))  # 128 -> 64

        x1d3 = ((Fun.relu(self.brk3(x1d2))))  # 64 -> 32
        x2d3 = ((Fun.relu(self.brk3(x2d2))))  # 64 -> 32
        x3d3 = ((Fun.relu(self.brk3(x3d2))))  # 64 -> 32
        x4d3 = ((Fun.relu(self.brk3(x4d2))))  # 64 -> 32

        x1d4 = ((Fun.relu(self.brk4(x1d3))))  # 32 -> 16
        x2d4 = ((Fun.relu(self.brk4(x2d3))))  # 32 -> 16
        x3d4 = ((Fun.relu(self.brk4(x3d3))))  # 32 -> 16
        x4d4 = ((Fun.relu(self.brk4(x4d3))))  # 32 -> 16

        # heads
        y1o = Fun.softmax(self.y1o(x1d4), dim=1)
        y2o = Fun.softmax(self.y2o(x2d4), dim=1)
        y3o = Fun.softmax(self.y3o(x3d4), dim=1)
        y4o = Fun.softmax(self.y4o(x4d4), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o


class CombineMultiOutputModelWeightedConcat(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModelWeightedConcat, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.x2 = nn.Linear(256, 256)  # 512; no because it is not for second input. It is to apply 2nd linear layer
        nn.init.xavier_normal_(self.x2.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.bn2 = nn.BatchNorm1d(256, eps=2e-1)
        self.bn3 = nn.BatchNorm1d(128, eps=2e-1)
        self.bn4 = nn.BatchNorm1d(64, eps=2e-1)

        self.x11 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x11.weight)

        self.x12 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x12.weight)

        self.x13 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x13.weight)

        self.x14 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x14.weight)

        self.x21 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x21.weight)

        self.x22 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x22.weight)

        self.x23 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x23.weight)

        self.x24 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x24.weight)

        # this is to apply linear layer after combination two vectors
        self.x1c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1c.weight)

        self.x2c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x2c.weight)

        self.x3c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x3c.weight)

        self.x4c = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x4c.weight)

        # heads
        self.y1o = nn.Linear(256, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  # initialize the nodes with the initial weights
        self.y2o = nn.Linear(256, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(256, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(256, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        # network for dynamic weight
        self.xw11 = nn.Linear(512, 128)
        nn.init.xavier_normal_(self.xw11.weight)

        self.xw12 = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.xw12.weight)

        self.xw13 = nn.Linear(64, 64)
        nn.init.xavier_normal_(self.xw13.weight)

        self.y1 = nn.Linear(64, 2)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y1.weight)

        self.xw21 = nn.Linear(512, 128)
        nn.init.xavier_normal_(self.xw21.weight)

        self.xw22 = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.xw22.weight)

        self.xw23 = nn.Linear(64, 64)
        nn.init.xavier_normal_(self.xw23.weight)

        self.y2 = nn.Linear(64, 2)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y2.weight)

        self.xw31 = nn.Linear(512, 128)
        nn.init.xavier_normal_(self.xw31.weight)

        self.xw32 = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.xw32.weight)

        self.xw33 = nn.Linear(64, 64)
        nn.init.xavier_normal_(self.xw33.weight)

        self.y3 = nn.Linear(64, 2)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y3.weight)

        self.xw41 = nn.Linear(512, 128)
        nn.init.xavier_normal_(self.xw41.weight)

        self.xw42 = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.xw42.weight)

        self.xw43 = nn.Linear(64, 64)
        nn.init.xavier_normal_(self.xw43.weight)

        self.y4 = nn.Linear(64, 2)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4.weight)

        self.d_out = nn.Dropout(0.5)

    def forward(self, x1_input, x2_input):
        x1_out = self.resnet_model(x1_input)
        x1_out = self.bn1(Fun.relu(self.x1(x1_out)))
        x1_out = self.bn2(Fun.relu(self.x2(x1_out)))

        x2_out = self.resnet_model(x2_input)
        x2_out = self.bn1(Fun.relu(self.x1(x2_out)))
        x2_out = self.bn2(Fun.relu(self.x2(x2_out)))

        x11 = self.bn1(Fun.relu(self.x11(x1_out)))
        x12 = self.bn1(Fun.relu(self.x12(x1_out)))
        x13 = self.bn1(Fun.relu(self.x13(x1_out)))
        x14 = self.bn1(Fun.relu(self.x14(x1_out)))

        x21 = self.bn1(Fun.relu(self.x21(x2_out)))
        x22 = self.bn1(Fun.relu(self.x22(x2_out)))
        x23 = self.bn1(Fun.relu(self.x23(x2_out)))
        x24 = self.bn1(Fun.relu(self.x24(x2_out)))

        # dynamic weight
        xw11 = self.bn3(Fun.relu(self.xw11(torch.cat((x11, x21), dim=1))))
        xw12 = self.bn4(Fun.relu(self.xw12(xw11)))
        xw13 = self.bn4(Fun.relu(self.xw13(xw12)))
        y1 = Fun.softmax(self.y1(xw13), dim=1)

        xw21 = self.bn3(Fun.relu(self.xw21(torch.cat((x12, x22), dim=1))))
        xw22 = self.bn4(Fun.relu(self.xw22(xw21)))
        xw23 = self.bn4(Fun.relu(self.xw23(xw22)))
        y2 = Fun.softmax(self.y2(xw23), dim=1)

        xw31 = self.bn3(Fun.relu(self.xw31(torch.cat((x13, x23), dim=1))))
        xw32 = self.bn4(Fun.relu(self.xw32(xw31)))
        xw33 = self.bn4(Fun.relu(self.xw33(xw32)))
        y3 = Fun.softmax(self.y3(xw33), dim=1)

        xw41 = self.bn3(Fun.relu(self.xw41(torch.cat((x14, x24), dim=1))))
        xw42 = self.bn4(Fun.relu(self.xw42(xw41)))
        xw43 = self.bn4(Fun.relu(self.xw43(xw42)))
        y4 = Fun.softmax(self.y4(xw43), dim=1)

        for col_1 in range(x11.shape[1]):
            x11[:, [col_1]] = x11[:, [col_1]] * y1[:, [0]]
            x21[:, [col_1]] = x21[:, [col_1]] * y1[:, [1]]

            x12[:, [col_1]] = x12[:, [col_1]] * y2[:, [0]]
            x22[:, [col_1]] = x22[:, [col_1]] * y2[:, [1]]

            x13[:, [col_1]] = x13[:, [col_1]] * y3[:, [0]]
            x23[:, [col_1]] = x23[:, [col_1]] * y3[:, [1]]

            x14[:, [col_1]] = x14[:, [col_1]] * y4[:, [0]]
            x24[:, [col_1]] = x24[:, [col_1]] * y4[:, [1]]

        x11 = torch.cat((x11, x21), 1)
        # print('The value of x11', x11)

        x12 = torch.cat((x12, x22), 1)
        # print('The value of x12', x12*(y2(2)))

        x13 = torch.cat((x13, x23), 1)
        # print('The value of x13', x13)

        x14 = torch.cat((x14, x24), 1)
        # print('The value of x14', x14)

        x1c = self.d_out(self.bn1(Fun.relu(self.x1c(x11))))
        x2c = self.d_out(self.bn1(Fun.relu(self.x2c(x12))))
        x3c = self.d_out(self.bn1(Fun.relu(self.x3c(x13))))
        x4c = self.d_out(self.bn1(Fun.relu(self.x4c(x14))))

        # heads
        y1o = Fun.softmax(self.y1o(x1c), dim=1)
        y2o = Fun.softmax(self.y2o(x2c), dim=1)
        y3o = Fun.softmax(self.y3o(x3c), dim=1)
        y4o = Fun.softmax(self.y4o(x4c), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o
