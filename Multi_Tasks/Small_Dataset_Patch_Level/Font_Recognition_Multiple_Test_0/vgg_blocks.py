import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()

        self.triple_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        return self.triple_conv(x)


class FourthConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourthConv, self).__init__()

        self.fourth_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        return self.fourth_conv(x)


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first_l=False, final_l=False):
        super(FullyConnectedLayer, self).__init__()

        self.first_l = first_l
        self.final_l = final_l

        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        if self.first_l:
            x1 = torch.flatten(x)
            return self.linear(x1)

        elif self.final_l:
            x1 = self.linear(x)
            return nn.Softmax(dim=-1)(x1)

        else:
            return self.linear(x)


class VggFlatAvgPool(nn.Module):

    def __init__(self):
        super(VggFlatAvgPool, self).__init__()

        self.flat_avg_pool = nn.AdaptiveAvgPool1d(6)
        self.just_flatten = nn.Flatten()

    def forward(self, x):
        x_out = self.flat_avg_pool(x)
        x_out = self.just_flatten(x_out)

        return x_out


class VggFeatureLinearLayer(nn.Module):

    def __init__(self):
        super(VggFeatureLinearLayer, self).__init__()

        self.classifier = nn.Sequential(

            # FC 1
            nn.Dropout(p=0.5),
            nn.Linear(512 * 6, 768),
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