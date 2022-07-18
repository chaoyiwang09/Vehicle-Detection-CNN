import os
import numpy as np
import librosa
import torch.nn as nn
import python_speech_features
import pandas as pd
import random

def get_mfcc(data, fs, winSize, winStep):
    # MFCC
    wav_feature = python_speech_features.mfcc(data, fs,
    numcep=13, winlen=winSize, winstep=winStep, nfilt=26, nfft=1600, lowfreq=0, highfreq=None, preemph=0.97)
    d_mfcc_feat = python_speech_features.delta(wav_feature, 1)
    d_mfcc_feat2 = python_speech_features.delta(wav_feature, 2)
    feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    return feature



train_file_list_1 = os.listdir("autodl-tmp/trainVehicle/")
random.shuffle(train_file_list_1)
print("Over all ",len(train_file_list_1), " files")

dfMFCC_final = pd.DataFrame()
file_count = 1
for i in train_file_list_1[:-90]:
    if file_count%30 == 0:
        print(file_count,"th files")
    trainData = pd.read_csv("autodl-tmp/trainVehicle/"+i,sep = "\s+",header = None)
    mfcc = pd.DataFrame(get_mfcc(np.array(trainData[1], dtype=float),8000,0.2,0.2))
    dfMFCC_final = pd.concat([dfMFCC_final,mfcc],ignore_index = True)
    file_count = file_count + 1

dfMFCC_final.to_csv("FeatureTrain1.csv",header=None, index = False)

dfMFCC_final = pd.DataFrame()
file_count = 1
for i in train_file_list_1[-90:]:
    if file_count%30 == 0:
        print(file_count,"th files")
    trainData = pd.read_csv("autodl-tmp/trainVehicle/"+i,sep = "\s+",header = None)
    mfcc = pd.DataFrame(get_mfcc(np.array(trainData[1], dtype=float),8000,0.2,0.2))
    dfMFCC_final = pd.concat([dfMFCC_final,mfcc],ignore_index = True)
    file_count = file_count + 1

dfMFCC_final.to_csv("FeatureTest1.csv",header=None, index = False)

dfMFCC_final = pd.DataFrame()
train_file_list_0 = os.listdir("autodl-tmp/trainPerson/")
random.shuffle(train_file_list_0)
print("Over all ",len(train_file_list_0), " files")
file_count = 1
dfMFCC_final = pd.DataFrame()
for i in train_file_list_0[:-20]:
    if file_count%30 == 0:
        print(file_count,"th files")
    trainData = pd.read_csv("autodl-tmp/trainPerson/"+i,sep = "\s+",header = None)
    mfcc = pd.DataFrame(get_mfcc(np.array(trainData[1], dtype=float),8000,0.2,0.2))
    dfMFCC_final = pd.concat([dfMFCC_final,mfcc],ignore_index = True)
    file_count = file_count + 1

dfMFCC_final.to_csv("FeatureTrain0.csv",header=None, index = False)

file_count = 1
dfMFCC_final = pd.DataFrame()
for i in train_file_list_0[-20:]:
    if file_count%30 == 0:
        print(file_count,"th files")
    trainData = pd.read_csv("autodl-tmp/trainPerson/"+i,sep = "\s+",header = None)
    mfcc = pd.DataFrame(get_mfcc(np.array(trainData[1], dtype=float),8000,0.2,0.2))
    dfMFCC_final = pd.concat([dfMFCC_final,mfcc],ignore_index = True)
    file_count = file_count + 1

dfMFCC_final.to_csv("FeatureTest0.csv",header=None, index = False)
