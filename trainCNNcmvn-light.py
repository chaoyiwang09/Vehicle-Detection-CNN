import os
import numpy as np
import librosa
import torch.nn as nn
import python_speech_features
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import myCNN
import warnings
from pytorch_model_summary import summary
warnings.filterwarnings('ignore')
class AcousticDataset(Dataset):
    def __init__(self,root_dir):
        self.data = pd.read_csv(root_dir,sep = ",",header = None)
        
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,index):
        global train_feat_mean
        global train_feat_std
        if self.data.iloc[index][39] == 1:
            label = [1,0]
        else:
            label = [0,1]
        feats = ( self.data.iloc[index][:39] - train_feat_mean )/train_feat_std
        return torch.FloatTensor(self.data.iloc[index][:39].values).resize(1,39),torch.FloatTensor(label)


# CMVN
train_feat = pd.read_csv("FeatureTrain.csv",sep = ",",header = None)
test_feat = pd.read_csv("FeatureTest.csv",sep = ",",header = None)    
train_feat_mean = train_feat[:39].mean()
train_feat_std = train_feat[:39].std()

trainData = AcousticDataset("FeatureTrain.csv")
testData = AcousticDataset("FeatureTest.csv")
train_dataloader = DataLoader(trainData,batch_size = 512,shuffle=True)
test_dataloader = DataLoader(testData,batch_size = 20000)

device = torch.device("cuda")
# Create Network
myNet = myCNN.myCNNnet3()
print(summary(myNet,torch.zeros((1,1, 39))))

myNet = myNet.to(device)

# Loss Function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

#learningRate = 0.01
#optimizer = torch.optim.SGD(myNet.parameters(),lr = learningRate)

total_train_step = 0
total_test_step = 0

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("tf-logs")
import time

start_time = time.time()
epoch = 100
for i in range(epoch):
    if i <30:
        learningRate = 0.01
        optimizer = torch.optim.SGD(myNet.parameters(),lr = learningRate)

    if i >=30:
        learningRate = 0.01
        optimizer = torch.optim.SGD(myNet.parameters(),lr = learningRate)
    
    print("{}th training start".format(i+1))
    
    # Start training
    myNet.train()
    for data in train_dataloader:
        features, targets = data
        features = features.to(device)
        targets = targets.to(device)
        outputs = myNet(features)
        loss = loss_fn(outputs,targets)
        
        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step = total_train_step + 1
        if total_train_step%500 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("{}th training, Loss:{}".format(total_train_step,loss))
        writer.add_scalar("train_loss", loss.item(),total_train_step)
            
    # Validating
    myNet.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            features, targets = data
            features = features.to(device)
            targets = targets.to(device)
            outputs = myNet(features)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets.argmax(1)).sum()
            total_accuracy = total_accuracy + accuracy
            print("test accuracy", accuracy)
    print("total_accuracy", total_accuracy/20000)    
    print("total loss of test data: {}".format(total_test_loss))
    writer.add_scalar("test_loss", loss.item(),total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/20000,total_test_step)
    total_test_step = total_test_step + 1    
    
    torch.save(myNet, "myNet_{}.pth".format(i))
    print("model saved")
    
