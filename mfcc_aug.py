import os
import numpy as np
import librosa
import torch.nn as nn
import python_speech_features
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

mean_mel = 1.9327990
std_mel = 2.6581813

def melAugment(data, fs, winSize, winStep):
    # MFCC
    wav_feature = python_speech_features.mfcc(data, fs,
                  numcep=13, winlen=winSize, winstep=winStep,  # winstep = 0.01
                  nfilt=26, nfft=1600, lowfreq=0, highfreq=None, preemph=0.97)
    #print(wav_feature)
    #print(wav_feature.shape)
    
    # Convert MFCC back to mel
    mel = librosa.feature.inverse.mfcc_to_mel(wav_feature.transpose())
    mask_area = random.randint(5,8)
    start_point =random.randint(1,mel.shape[0]-mask_area)
    #mel[start_point:mask_area+start_point,:] = np.random.randn()*std_mel + mean_mel
    mask_value = np.random.randn()*std_mel + mean_mel
    while mask_value < 0:
        mask_value = np.random.randn()*std_mel + mean_mel
    mel[start_point:mask_area+start_point,:] = mask_value
    #print(np.random.randn()*std_mel + mean_mel)
    #print(mel)
    #print(mel.shape)
    
    # Compute log and do DCT to get MFCC
    try:
        mel = 10*np.log10(mel)
    except:
        print("Use original mel")
        mel = librosa.feature.inverse.mfcc_to_mel(wav_feature.transpose())
    wav_feature = librosa.feature.mfcc(y=None,sr=8000,S=mel,n_mfcc=13,dct_type=2, norm='ortho', lifter=0)
    wav_feature = wav_feature.transpose()
    #print(type(wav_feature))
    # diff1, diff2
    d_mfcc_feat = python_speech_features.delta(wav_feature, 1)
    d_mfcc_feat2 = python_speech_features.delta(wav_feature, 2)
    feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    # size 39
    return feature
'''
train_file_list_1 = os.listdir("autodl-tmp/trainVehicle/")
random.shuffle(train_file_list_1)
print("Over all ",len(train_file_list_1), " files")

dfMFCC_final = pd.DataFrame()
file_count = 1
for i in train_file_list_1[:-90]:
    if file_count%30 == 0:
        print(file_count,"th files")
    trainData = pd.read_csv("autodl-tmp/trainVehicle/"+i,sep = "\s+",header = None)
    #mfcc1 = pd.DataFrame(get_mfcc(np.array(trainData[1], dtype=float),8000,0.2,0.2))
    mfcc = pd.DataFrame(melAugment(np.array(trainData[1], dtype=float),8000,0.2,0.2))
    #print(mfcc1.head(5))
    #print(mfcc2.head(5))
    dfMFCC_final = pd.concat([dfMFCC_final,mfcc],ignore_index = True)
    file_count = file_count + 1

dfMFCC_final.to_csv("FeatureTrain1Aug.csv",header=None, index = False)
'''
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
    mfcc = pd.DataFrame(melAugment(np.array(trainData[1], dtype=float),8000,0.2,0.2))
    dfMFCC_final = pd.concat([dfMFCC_final,mfcc],ignore_index = True)
    file_count = file_count + 1

dfMFCC_final.to_csv("FeatureTrain0.csv",header=None, index = False)
