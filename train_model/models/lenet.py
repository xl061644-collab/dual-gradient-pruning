import torch
import torch.nn as nn

from src.VariationalBottleneck import VariationalBottleneck

# import random
# random.seed(1234)
class LeNetZhu(torch.nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = torch.nn.Sigmoid
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            torch.nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = torch.nn.Sequential(torch.nn.Linear(768, num_classes))
        # self.feature=None
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        # self.feature=out
        out = self.fc(out)
        return out




class _Select(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, : self.n]


class ModifiedBlock(torch.nn.Module):
    def __init__(self, old_Block):
        super().__init__()
        self.attn = old_Block.attn
        self.drop_path = old_Block.drop_path
        self.norm2 = old_Block.norm2
        self.mlp = old_Block.mlp

    def forward(self, x):
        x = self.attn(x)
        x = self.drop_path(self.mlp((self.norm2(x))))
        return x


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Sequential(     #input_size=(1*28*28)
#             nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
#             nn.ReLU(),      #input_size=(6*28*28)
#             nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(6, 16, 5),
#             nn.ReLU(),      #input_size=(16*10*10)
#             nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(16 * 5 * 5, 120),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(120, 84),
#             nn.ReLU()
#         )
#         self.fc3 = nn.Linear(84, 10)
#
#     # 定义前向传播过程，输入为x
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

def lenet(num_classes=10, channels=3):
    return LeNetZhu(num_channels=channels, num_classes=num_classes)

# def lenet1(num_classes=10, channels=1):
#     return  LeNet()