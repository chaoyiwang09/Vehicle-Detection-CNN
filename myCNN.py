import torch.nn as nn

class myCNNnet(nn.Module):  # create myDNN model
    def __init__(self):
        super(myCNNnet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1,32,3,stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.Conv1d(32,32,3,stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(448,16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16,2),
        )

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.model(x)
        return x
    
class myCNNnet2(nn.Module):  # create myDNN model
    def __init__(self):
        super(myCNNnet2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1,16,3,stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,3,stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,16,3,stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(144,16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16,2),
        )

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.model(x)
        return x
    
class myCNNnet3(nn.Module):  # create myDNN model
    def __init__(self):
        super(myCNNnet3, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1,1,3,stride=1,groups = 1, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Conv1d(1,16,1,1,0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,16,3,stride=1,groups = 16, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16,32,1,1,0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,32,3,stride=1,groups = 32, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,16,1,1,0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(144,16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16,2),
        )

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.model(x)
        return x
    
class myCNNnet4(nn.Module):  # create myDNN model
    def __init__(self):
        super(myCNNnet4, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1,1,3,stride=1,groups = 1, padding=1),
            nn.Conv1d(1,16,1,1,0),
            nn.MaxPool1d(2),
            nn.Conv1d(16,16,3,stride=1,groups = 16, padding=1),
            nn.Conv1d(16,32,1,1,0),
            nn.MaxPool1d(2),
            nn.Conv1d(32,32,3,stride=1,groups = 32, padding=1),
            nn.Conv1d(32,16,1,1,0),
            nn.Flatten(),
            nn.Linear(144,16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16,2),
        )

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.model(x)
        return x
