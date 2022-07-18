import torch.nn as nn

class myDNNnet(nn.Module):  # create myDNN model
    def __init__(self):
        super(myDNNnet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(39, 64),
            nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            nn.Linear(32, 8),
            nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            nn.Linear(8, 2),
        )

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.model(x)
        return x