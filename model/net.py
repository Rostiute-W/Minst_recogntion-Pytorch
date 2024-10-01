import torch
import torch.nn as nn

# -------------------------------------------------------------#
#                      自定义网络结构
# -------------------------------------------------------------#

class MinstNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MinstNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    import torchsummary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.ones(1, 3, 28, 28).to(device)
    net = MinstNet().to(device)
    print(net)
    output = net(input)
    print(output.shape)
    torchsummary.summary(net, input_size=(3, 28, 28))