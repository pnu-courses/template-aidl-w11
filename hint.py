"""제11주 참고 코드 — 05 · 06 (모델 전체) 두 모델의 PyTorch 정의 (7주차와 같다 · 수정 금지 · 실행하지 않아도 된다)

채점 환경에는 torch 가 없습니다. 구조와 weights/*.npz 의 이름을 확인하는 용도입니다.
두 모델 모두 CIFAR-10 (RGB 32x32) 으로 학습했고, 입력은 (img / 255 - CIFAR_MEAN) / CIFAR_STD 입니다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------- ResNet18 (제3주 구조, 입력 3 채널)

def downsample(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = shortcut

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)   # 64x16x16
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                          # 64x8x8
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))                       # 64x8x8
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2, downsample(64, 128, 2)), BasicBlock(128, 128))     # 4x4
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2, downsample(128, 256, 2)), BasicBlock(256, 256))   # 2x2
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2, downsample(256, 512, 2)), BasicBlock(512, 512))   # 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# ----------------------------------------------------------------- MobileNetV2 (CIFAR-10)

class ConvBN(nn.Module):
    """conv (패딩 (k-1)/2, 편향 없음) -> BN -> ReLU6 (act=False 면 활성화 없음)."""

    def __init__(self, cin, cout, k, stride=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu6(x) if self.act else x


class InvertedResidual(nn.Module):
    """1x1 확장 (t 배, ReLU6) -> 3x3 depthwise (ReLU6) -> 1x1 축소 (활성화 없음) [+ 입력]."""

    def __init__(self, cin, cout, stride, t):
        super().__init__()
        hidden = cin * t
        self.use_res = stride == 1 and cin == cout
        self.expand = ConvBN(cin, hidden, 1) if t != 1 else None
        self.dw = ConvBN(hidden, hidden, 3, stride, groups=hidden)
        self.project = ConvBN(hidden, cout, 1, act=False)

    def forward(self, x):
        y = self.expand(x) if self.expand is not None else x
        y = self.project(self.dw(y))
        return x + y if self.use_res else y


class MobileNetV2(nn.Module):
    # (t, c, n, s): 확장 배수, 출력 채널, 반복 수, 첫 블록의 stride. 32x32 입력이라 stem 과 두 번째 단계는 stride 1
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = ConvBN(3, 32, 3, 1)                                       # 32x32x32
        blocks, cin = [], 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                blocks.append(InvertedResidual(cin, c, s if i == 0 else 1, t))
                cin = c
        self.blocks = nn.ModuleList(blocks)                                   # blocks.0 ... blocks.16 -> 320x4x4
        self.head = ConvBN(320, 1280, 1)                                      # 1280x4x4
        self.fc = nn.Linear(1280, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        x = self.head(x)
        return self.fc(x.mean((2, 3)))


MODELS = {"resnet18": ResNet18, "mobilenetv2": MobileNetV2}
