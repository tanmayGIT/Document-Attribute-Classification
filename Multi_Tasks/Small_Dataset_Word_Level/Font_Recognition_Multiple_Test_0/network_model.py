from __future__ import print_function
from __future__ import division
import torch.nn as nn

from vgg_blocks import *
import numpy as np

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

            model_ft.fc = nn.Linear(num_ftrs, 512)
            input_size = 224

        elif model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet34(pretrained=use_pretrained)
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
        

class MultiOutputModel(nn.Module):
    def __init__(self, model_core, dd):
        super(MultiOutputModel, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.x2 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps=2e-1)

        # heads
        self.y1o = nn.Linear(256, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  #
        self.y2o = nn.Linear(256, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(256, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(256, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(dd)

    def forward(self, x):
        x1 = self.resnet_model(x)

        x1 = self.bn1(Fun.relu(self.x1(x1)))
        x1 = self.bn2(Fun.relu(self.x2(x1)))

        # heads
        y1o = Fun.softmax(self.y1o(x1))
        y2o = Fun.softmax(self.y2o(x1))
        y3o = Fun.softmax(self.y3o(x1))
        y4o = Fun.softmax(self.y4o(x1))  # should be sigmoid

        return y1o, y2o, y3o, y4o


class MultiOutputModelConvAlex(nn.Module):
    def __init__(self, model_core, dd):
        super(MultiOutputModelConvAlex, self).__init__()

        self.resnet_model = model_core

        self.Conv_1 = ConvBNRelu(1, 64, 11, 4, 2)
        self.Conv_2 = ConvBNRelu(64, 192, 5, 1, 2)
        self.Conv_3 = ConvBNRelu(192, 384, 3, 1, 1)
        self.Conv_4 = ConvBNRelu(384, 256, 3, 1, 1)
        self.Conv_5 = ConvBNRelu(256, 256, 3, 1, 1)

        self.AvgPoolFlat = FlatAvgPool()
        self.Last_Linear_Layer = FeatureLinearLayer()

        # heads
        self.y1o = nn.Linear(192, 3)  # this is for the number of scanning class
        nn.init.xavier_normal_(self.y1o.weight)  #
        self.y2o = nn.Linear(192, 3)  # this is for number of font size class
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(192, 6)  # this is for number of font type class
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(192, 4)  # this is for number of font emphasis class
        nn.init.xavier_normal_(self.y4o.weight)

        self.d_out = nn.Dropout(dd)

    def forward(self, x):
        x1_out = self.resnet_model(x)  # 2048 features

        # generate 2 channels from these two 1D vectors
        n_batch_size = x1_out.shape[0]
        n_rows = x1_out.shape[1]

        combine_features = np.empty((n_batch_size, 1, n_rows))  # n_batch, nChannel, img_rows, img_cols
        combine_features = torch.from_numpy(combine_features).float()
        combine_features = torch.tensor(combine_features)
        # making 2D vector into 3D vector
        for iBatch in range(0, n_batch_size):
            combine_features[iBatch, 0, :] = x1_out[iBatch, :]  # making 1st channel

        combine_features = combine_features.cuda()
        x_conv_out = self.Conv_1(combine_features)
        x_conv_out = self.Conv_2(x_conv_out)
        x_conv_out = self.Conv_3(x_conv_out)
        x_conv_out = self.Conv_4(x_conv_out)
        x_conv_out = self.Conv_5(x_conv_out)

        linear_features = self.AvgPoolFlat(x_conv_out)
        linear_features = self.Last_Linear_Layer(linear_features)

        # heads
        y1o = Fun.softmax(self.y1o(linear_features))
        y2o = Fun.softmax(self.y2o(linear_features))
        y3o = Fun.softmax(self.y3o(linear_features))
        y4o = Fun.softmax(self.y4o(linear_features))  # should be sigmoid

        return y1o, y2o, y3o, y4o


class MultiOutputModelConvVggNet(nn.Module):
    def __init__(self, model_core, dd):
        super(MultiOutputModelConvVggNet, self).__init__()

        self.resnet_model = model_core

        self.double_conv_1 = DoubleConv(1, 64)
        self.double_conv_2 = DoubleConv(64, 128)

        self.fourth_conv_1 = FourthConv(128, 256)
        self.fourth_conv_2 = FourthConv(256, 512)
        self.fourth_conv_3 = FourthConv(512, 512)

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

    def forward(self, x):
        x1_out = self.resnet_model(x)  # 2048 features

        # generate 2 channels from these two 1D vectors
        n_batch_size = x1_out.shape[0]
        n_rows = x1_out.shape[1]

        combine_features = np.empty((n_batch_size, 1, n_rows))  # n_batch, nChannel, img_rows, img_cols
        combine_features = torch.from_numpy(combine_features).float()
        combine_features = torch.tensor(combine_features)
        # making 2D vector into 3D vector
        for iBatch in range(0, n_batch_size):
            combine_features[iBatch, 0, :] = x1_out[iBatch, :]  # making 1st channel

        combine_features = combine_features.cuda()
        x1 = self.double_conv_1(combine_features)
        x2 = self.double_conv_2(x1)

        x3 = self.fourth_conv_1(x2)
        x4 = self.fourth_conv_2(x3)
        x_conv_out = self.fourth_conv_3(x4)

        linear_features = self.AvgPoolFlat(x_conv_out)
        linear_features = self.Last_Linear_Layer(linear_features)

        # heads
        y1o = Fun.softmax(self.y1o(linear_features), dim=1)
        y2o = Fun.softmax(self.y2o(linear_features), dim=1)
        y3o = Fun.softmax(self.y3o(linear_features), dim=1)
        y4o = Fun.softmax(self.y4o(linear_features), dim=1)  # should be sigmoid

        return y1o, y2o, y3o, y4o

