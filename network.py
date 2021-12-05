"""
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, activation="swish"):
        super().__init__()
        activation_dict = {"relu": nn.ReLU(inplace=True), "swish": nn.SiLU(inplace=True)}
        self.act_fn = activation_dict[activation]
        self.shortcut = nn.Sequential()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.act_fn(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act_fn(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, activation="swish"):
        super().__init__()
        activation_dict = {"relu": nn.ReLU(inplace=True), "swish": nn.SiLU(inplace=True)}
        self.act_fn = activation_dict[activation]

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.act_fn(self.bn1(self.conv1(x)))
        out = self.act_fn(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act_fn(out)

        return out


class FerResNet(nn.Module):

    def __init__(self, block, num_blocks, n_filters=64, activation="swish", num_classes=7):
        super().__init__()
        self.in_channels = 3
        self.n_filters = n_filters

        activation_dict = {"relu": nn.ReLU(inplace=True), "swish": nn.SiLU(inplace=True)}
        self.act_fn = activation_dict[activation]
        self.avg_pool2d = nn.AvgPool2d(4)

        self.conv1 = nn.Conv2d(self.in_channels, n_filters, kernel_size=7, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, n_filters, num_blocks[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, n_filters * 2, num_blocks[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, n_filters * 4, num_blocks[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, n_filters * 8, num_blocks[3], stride=2, activation=activation)
        self.linear = nn.Linear(n_filters * 8 * block.expansion, num_classes)

    def _make_layer(self, block, n_filters, num_blocks, stride, activation):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = nn.ModuleList()
        for stride in strides:
            layers.append(block(self.n_filters, n_filters, stride, activation))
            self.n_filters = n_filters * block.expansion

        return nn.Sequential(*layers).apply(init_weights)

    def forward(self, x):
        out = self.max_pool(self.act_fn(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    @staticmethod
    def print_model_summary(model, inputs):
        return summary(model.cuda(), inputs)

    @staticmethod
    def print_model(model):
        named_layers = dict(model.named_modules())
        print(f"Layers: {named_layers}")

    @staticmethod
    def model_capacity(model):
        num = 1
        conv_layer = {}
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                num += 1
                conv_layer[name] = layer

        number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        number_of_non_learnable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        number_of_layers = len(list(model.parameters()))
        print(f"Number of convolution layers: {num}")
        print(f"Total number of layers: {number_of_layers}")
        print(f"Number of trainable parameters: {number_of_learnable_params:,}")
        print(f"Number of non-trainable parameters: {number_of_non_learnable_params:,}")


def FerNet54():
    return FerResNet(Bottleneck, [3, 4, 6, 3], activation="swish")


def FerNet101():
    return FerResNet(Bottleneck, [3, 4, 22, 3], activation="swish")


def main():
    net = FerNet54()
    net.model_capacity(net)
    net.print_model(net)
    # net.print_model_summary(net, (3, 48, 48))


# ResNet18=BasicBlock[2, 2, 2, 2]
# ResNet34=BasicBlock[3, 4, 6, 3]
# ResNet50=Bottleneck[3, 4, 6, 3]
# ResNet101=Bottleneck[3, 4, 23, 3]
# ResNet18=Bottleneck152[3, 8, 36, 3]

if __name__ == "__main__":
    main()
