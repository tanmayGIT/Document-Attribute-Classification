from __future__ import print_function
from __future__ import division
import torch.nn as nn

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


class CombineMultiOutputModelConcat(nn.Module):
    def __init__(self, model_core, dd):
        super(CombineMultiOutputModelConcat, self).__init__()

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

        x11 = torch.cat((x11, x21), 1)
        # print('The value of x11', x11)

        x12 = torch.cat((x12, x22), 1)
        # print('The value of x12', x12)

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
