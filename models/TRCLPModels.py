# Copyright (c) Wang Zhimin zm.wang@buaa.edu.cn 2024/10/28 All Rights Reserved.


import torch
import torch.nn as nn
from math import floor
import numpy as np
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class comb_FC(nn.Module):
    def __init__(self, eyeFeatureSize, headFeatureSize, gwFeatureSize, numClasses):
        super().__init__()
        # eye-in-head features
        self.eyeFeatureSize = eyeFeatureSize
        # head features
        self.headFeatureSize = headFeatureSize
        # gaze-in-world features
        self.gwFeatureSize = gwFeatureSize
        
        # preset params
        self.eyeFeatureNum = 2
        self.eyeFeatureLength = int(self.eyeFeatureSize/self.eyeFeatureNum)
        self.headFeatureNum = 2
        self.headFeatureLength = int(self.headFeatureSize/self.headFeatureNum)
        self.gwFeatureNum = 2
        self.gwFeatureLength = int(self.gwFeatureSize/self.gwFeatureNum)
        
        # Eye_CNN1D Module
        eyeCNN1D_outChannels1 = 16
        eyeCNN1D_kernelSize1 = 3
        eyeCNN1D_featureSize1 = floor((self.eyeFeatureLength - eyeCNN1D_kernelSize1 + 1)/2)
        eyeCNN1D_outChannels2 = 16
        eyeCNN1D_kernelSize2 = 3
        eyeCNN1D_featureSize2 = floor((eyeCNN1D_featureSize1 - eyeCNN1D_kernelSize2 + 1)/2)
        eyeCNN1D_outChannels3 = 16
        eyeCNN1D_kernelSize3 = 3
        eyeCNN1D_featureSize3 = floor((eyeCNN1D_featureSize2 - eyeCNN1D_kernelSize3 + 1)/2)
        self.eyeCNN1D_outputSize = eyeCNN1D_featureSize3 * eyeCNN1D_outChannels3
        self.eyeCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.eyeFeatureNum, out_channels=eyeCNN1D_outChannels1,kernel_size=eyeCNN1D_kernelSize1),
            nn.BatchNorm1d(eyeCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=eyeCNN1D_outChannels1, out_channels=eyeCNN1D_outChannels2,kernel_size=eyeCNN1D_kernelSize2),
            nn.BatchNorm1d(eyeCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=eyeCNN1D_outChannels2, out_channels=eyeCNN1D_outChannels3,kernel_size=eyeCNN1D_kernelSize3),
            nn.BatchNorm1d(eyeCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
        
        # Eye_GRU Module
        self.eyeGRU_hidden_size = 64
        self.eyeGRU_layers = 1
        self.eyeGRU_directions = 2
        self.eyeGRU = nn.GRU(eyeCNN1D_outChannels3,self.eyeGRU_hidden_size, self.eyeGRU_layers, batch_first=True, bidirectional=bool(self.eyeGRU_directions-1))
        
        # Head_CNN1D Module
        headCNN1D_outChannels1 = 16
        headCNN1D_kernelSize1 = 3
        headCNN1D_featureSize1 = floor((self.headFeatureLength - headCNN1D_kernelSize1 + 1)/2)
        headCNN1D_outChannels2 = 16
        headCNN1D_kernelSize2 = 3
        headCNN1D_featureSize2 = floor((headCNN1D_featureSize1 - headCNN1D_kernelSize2 + 1)/2)
        headCNN1D_outChannels3 = 16
        headCNN1D_kernelSize3 = 3
        headCNN1D_featureSize3 = floor((headCNN1D_featureSize2 - headCNN1D_kernelSize3 + 1)/2)
        self.headCNN1D_outputSize = headCNN1D_featureSize3 * headCNN1D_outChannels3
        self.headCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headCNN1D_outChannels1,kernel_size=headCNN1D_kernelSize1),
            nn.BatchNorm1d(headCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=headCNN1D_outChannels1, out_channels=headCNN1D_outChannels2,kernel_size=headCNN1D_kernelSize2),
            nn.BatchNorm1d(headCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=headCNN1D_outChannels2, out_channels=headCNN1D_outChannels3,kernel_size=headCNN1D_kernelSize3),
            nn.BatchNorm1d(headCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),       
             )
        
        # Head_GRU Module
        self.headGRU_hidden_size = 64
        self.headGRU_layers = 1
        self.headGRU_directions = 2
        self.headGRU = nn.GRU(headCNN1D_outChannels3,self.headGRU_hidden_size, self.headGRU_layers, batch_first=True, bidirectional=bool(self.headGRU_directions-1))
        
        # GW_CNN1D Module
        gwCNN1D_outChannels1 = 16
        gwCNN1D_kernelSize1 = 3
        gwCNN1D_featureSize1 = floor((self.gwFeatureLength - gwCNN1D_kernelSize1 + 1)/2)
        gwCNN1D_outChannels2 = 16
        gwCNN1D_kernelSize2 = 3
        gwCNN1D_featureSize2 = floor((gwCNN1D_featureSize1 - gwCNN1D_kernelSize2 + 1)/2)
        gwCNN1D_outChannels3 = 16
        gwCNN1D_kernelSize3 = 3
        gwCNN1D_featureSize3 = floor((gwCNN1D_featureSize2 - gwCNN1D_kernelSize3 + 1)/2)
        self.gwCNN1D_outputSize = gwCNN1D_featureSize3 * gwCNN1D_outChannels3
        self.gwCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gwFeatureNum, out_channels=gwCNN1D_outChannels1,kernel_size=gwCNN1D_kernelSize1),
            nn.BatchNorm1d(gwCNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=gwCNN1D_outChannels1, out_channels=gwCNN1D_outChannels2,kernel_size=gwCNN1D_kernelSize2),
            nn.BatchNorm1d(gwCNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=gwCNN1D_outChannels2, out_channels=gwCNN1D_outChannels3,kernel_size=gwCNN1D_kernelSize3),
            nn.BatchNorm1d(gwCNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),
             )
        
        # GW_GRU Module
        self.gwGRU_hidden_size = 64
        self.gwGRU_layers = 1
        self.gwGRU_directions = 2
        self.gwGRU = nn.GRU(gwCNN1D_outChannels3,self.gwGRU_hidden_size, self.gwGRU_layers, batch_first=True, bidirectional=bool(self.gwGRU_directions-1))
        
        # task prediction FC Module
        eyeGRU_output_size = self.eyeGRU_hidden_size*self.eyeGRU_directions
        headGRU_output_size = self.headGRU_hidden_size*self.headGRU_directions
        gwGRU_output_size = self.gwGRU_hidden_size*self.gwGRU_directions
        prdFC_inputSize = eyeGRU_output_size + headGRU_output_size + gwGRU_output_size
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numClasses
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward1(self, x):
        out = self.PrdFC(x)
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out
 

class sinCNN_BiGRU(nn.Module):
    def __init__(self, featureSize, featureIndex):
        super().__init__()
        # features
        self.featureSize = featureSize
        self.featureIndex = featureIndex
       
        print('feature size: {}'.format(self.featureSize))
        
        # preset params
        self.featureNum = 2
        self.featureLength = int(self.featureSize/self.featureNum)

        
        # CNN1D Module
        CNN1D_outChannels1 = 16
        CNN1D_kernelSize1 = 3
        CNN1D_featureSize1 = floor((self.featureLength - CNN1D_outChannels1 + 1)/2)
        CNN1D_outChannels2 = 16
        CNN1D_kernelSize2 = 3
        CNN1D_featureSize2 = floor((CNN1D_featureSize1 - CNN1D_kernelSize2 + 1)/2)
        CNN1D_outChannels3 = 16
        CNN1D_kernelSize3 = 3
        CNN1D_featureSize3 = floor((CNN1D_featureSize2 - CNN1D_kernelSize3 + 1)/2)
        self.CNN1D_outputSize = CNN1D_featureSize3 * CNN1D_outChannels3
        self.CNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.featureNum, out_channels=CNN1D_outChannels1,kernel_size=CNN1D_kernelSize1),
            nn.BatchNorm1d(CNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=CNN1D_outChannels1, out_channels=CNN1D_outChannels2,kernel_size=CNN1D_kernelSize2),
            nn.BatchNorm1d(CNN1D_outChannels2),
            nn.ReLU(),
            nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=CNN1D_outChannels2, out_channels=CNN1D_outChannels3,kernel_size=CNN1D_kernelSize3),
            nn.BatchNorm1d(CNN1D_outChannels3),
            nn.ReLU(),
            nn.MaxPool1d(2),    
             )
        
        # Eye_GRU Module
        self.GRU_hidden_size = 64
        self.GRU_layers = 1
        self.GRU_directions = 2
        self.GRU = nn.GRU(CNN1D_outChannels3,self.GRU_hidden_size, self.GRU_layers, batch_first=True, bidirectional=bool(self.GRU_directions-1))

        
    def forward1(self, x):

        feature = x[:, self.featureIndex * self.featureSize:(self.featureIndex+1) * self.featureSize]
        feature = feature.reshape(-1, self.featureLength, self.featureNum) # [256, 250, 2]
        feature = feature.permute(0,2,1)        # [256, 2, 250]

        featureOut = self.CNN1D(feature)     
        featureOut = featureOut.permute(0,2,1)     
        h0 = torch.zeros(self.GRU_layers*self.GRU_directions, x.size(0), self.GRU_hidden_size).to(device) 
        GruOut, _ = self.GRU(featureOut, h0)

        GruOut = GruOut[:, -1, :]
        
         
        return GruOut
            
    def forward(self, x):
        out = self.forward1(x)
        out = torch.nn.functional.normalize(out, dim=1)
        return out
