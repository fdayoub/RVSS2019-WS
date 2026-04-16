import torch.nn as nn
import torch.nn.functional as F


class SteerNet(nn.Module):
    def __init__(self, mode="regression"):
        super(SteerNet, self).__init__()
        if mode not in {"regression", "classification"}:
            raise ValueError("mode must be one of {'regression','classification'}")

        self.mode = mode
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1 if mode == "regression" else 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.mode == "classification":
            x = F.softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for shape in size:
            num_features *= shape
        return num_features


def test():
    import torch

    net = SteerNet(mode="regression")
    sample = torch.randn(2, 3, 84, 84)
    print("regression output shape:", net(sample).shape)

    net_class = SteerNet(mode="classification")
    print("classification output shape:", net_class(sample).shape)


if __name__ == "__main__":
    test()
