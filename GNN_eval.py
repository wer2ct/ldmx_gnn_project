#Imports
import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt 
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch_geometric.data import Data, Dataset
import os
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
import glob
from torch_geometric.nn import GCNConv as gcn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.nn import EdgeConv 
import csv
import sys
from torch_geometric.nn import global_mean_pool
import pandas as pd
from sklearn.metrics import roc_curve, auc

#Functions and Classes

#Latest functional GNN model. 
class GNN_v2(torch.nn.Module):
    def __init__(self, in_channels, hc1, hc2, hc3, fc1, fc2, fc3, out_channels): # we define all of the convolutions and layers here
        super(GNN_v2, self).__init__()
        
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, hc1),
            torch.nn.ReLU(),
            torch.nn.Linear(hc1, hc1)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hc1 * 2, hc2),
            torch.nn.ReLU(),
            torch.nn.Linear(hc2, hc2)
        ) 
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(hc2 * 2 , hc3),
            torch.nn.ReLU(),
            torch.nn.Linear(hc3, hc3)
        )  
        
        self.edgeconv1 = EdgeConv(self.mlp1, aggr='max')
        #self.bn1 = torch.nn.BatchNorm1d(hc1)
        
        self.edgeconv2 = EdgeConv(self.mlp2, aggr='max') 
        #self.bn2 = torch.nn.BatchNorm1d(hc2)
        
        self.edgeconv3 = EdgeConv(self.mlp3, aggr='max')
        self.bn3 = torch.nn.BatchNorm1d(hc3)
        
        self.linear1 = torch.nn.Linear(hc3, fc1) 
        self.linear2 = torch.nn.Linear(fc1, fc2)
        self.linear3 = torch.nn.Linear(fc2, fc3)
        self.head = torch.nn.Linear(fc3, out_channels) 

        self.parameter_init()

    def forward(self, x, edge_index, batch_idx): #then apply them in the forward defintion. 
        x = self.edgeconv1(x, edge_index) #apply first convolution. Here we are actually applying the inputs. 
        #x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.edgeconv2(x, edge_index)
        #x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.edgeconv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = global_mean_pool(x, batch_idx)

        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.linear3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.head(x)
        
        return(x) 

    def parameter_init(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu') #set weights to random from uniform weighted on Relu
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias) #set biases to zero (if there are any)


def run_inference(model, test_loader):
    model.eval()
    with torch.no_grad():
        i = 0
        total_correct = 0
        for data in test_loader:
            out = model(data.x, data.edge_index, data.batch)
            n_correct = torch.sum(out.argmax(dim=1) == data.y)
            i += len(data.y)
            total_correct += n_correct
    return(total_correct / i)

def main():
    #This script generates a ROC curve and GNN output plot for a given model 
    print("Beginning to load validation files")
    signal_val_file_list = []
    for path in signal_validation_paths:
        signal_val_file_list.append((torch.load(path, weights_only = False)))
        print(f"Loaded: {path}")

    #Now loading the background validation files
    val_background = torch.load(bkgs_validation_paths[0], weights_only = False)
    print("loaded validation files")

    #Finally, load our model
    trained_model_state_dict = torch.load('<path to state dict>', weights_only = True)
    model_preloaded = GNN_v2(in_channels = 4, hc1 = 20, hc2 = 40, hc3 = 50, fc1 = 25, fc2 = 12, fc3 = 6, out_channels = 2)
    model_preloaded.load_state_dict(trained_model_state_dict)

    #We need to declare a bunch of loaders here. This is terribly inefficient but I'm tired
    m050_bkgd_roc_sample = val_background[0]+val_background[1]+val_background[2]+val_background[3]+val_background[4]+ signal_val_file_list[0]
    m010_bkgd_roc_sample = val_background[0]+val_background[1]+val_background[2]+val_background[3]+val_background[4]+ signal_val_file_list[1]
    m100_bkgd_roc_sample = val_background[0]+val_background[1]+val_background[2]+val_background[3]+val_background[4]+ signal_val_file_list[2]
    m005_bkgd_roc_sample = val_background[0]+val_background[1]+val_background[2]+val_background[3]+val_background[4]+ signal_val_file_list[3]

    m010_roc_sample_loader = DataLoader(m010_bkgd_roc_sample, batch_size = 500, drop_last = True, shuffle=True, num_workers=1)
    m005_roc_sample_loader = DataLoader(m005_bkgd_roc_sample, batch_size = 500, drop_last = True, shuffle=True, num_workers=1)
    m100_roc_sample_loader = DataLoader(m100_bkgd_roc_sample, batch_size = 500, drop_last = True, shuffle=True, num_workers=1)
    m050_roc_sample_loader = DataLoader(m050_bkgd_roc_sample, batch_size = 500, drop_last = True, shuffle=True, num_workers=1)

    m010_gnnout_sample_loader = DataLoader(signal_val_file_list[1], batch_size = 500, drop_last = True, shuffle=True, num_workers=1)
    m005_gnnout_sample_loader = DataLoader(signal_val_file_list[3], batch_size = 500, drop_last = True, shuffle=True, num_workers=1)
    m100_gnnout_sample_loader = DataLoader(signal_val_file_list[2], batch_size = 500, drop_last = True, shuffle=True, num_workers=1)
    m050_gnnout_sample_loader = DataLoader(signal_val_file_list[0], batch_size = 500, drop_last = True, shuffle=True, num_workers=1)
    bkgd_gnnout_sample_loader = DataLoader(val_background[0]+val_background[1]+val_background[2]+val_background[3]+val_background[4], batch_size = 500, drop_last = True, shuffle=True, num_workers=1)

    #Create our fpr and tpr for ROC curve 
    y_scores_m010 = []
    y_true_m010 = []

    y_scores_m005 = []
    y_true_m005 = []

    y_scores_m100 = []
    y_true_m100 = []

    y_scores_m050 = []
    y_true_m050 = []


    with torch.no_grad():
        for data in m010_roc_sample_loader:
            # Move batch to GPU
            data = data.to(device)

            # Forward pass on GPU
            out = test_model_preloaded(data.x, data.edge_index, data.batch)

            # Assuming binary classification and out shape [N, 2]
            probs = torch.sigmoid(out)[:, 1]

            # Only CPU when appending
            y_scores_m010.append(probs.cpu())
            y_true_m010.append(data.y.cpu())

        for data in m005_roc_sample_loader:
            # Move batch to GPU
            data = data.to(device)

            # Forward pass on GPU
            out = test_model_preloaded(data.x, data.edge_index, data.batch)

            # Assuming binary classification and out shape [N, 2]
            probs = torch.sigmoid(out)[:, 1]

            # Only CPU when appending
            y_scores_m005.append(probs.cpu())
            y_true_m005.append(data.y.cpu())

        for data in m100_roc_sample_loader:
            # Move batch to GPU
            data = data.to(device)

            # Forward pass on GPU
            out = test_model_preloaded(data.x, data.edge_index, data.batch)

            # Assuming binary classification and out shape [N, 2]
            probs = torch.sigmoid(out)[:, 1]

            # Only CPU when appending
            y_scores_m100.append(probs.cpu())
            y_true_m100.append(data.y.cpu())

        for data in m050_roc_sample_loader:
            # Move batch to GPU
            data = data.to(device)

            # Forward pass on GPU
            out = test_model_preloaded(data.x, data.edge_index, data.batch)

            # Assuming binary classification and out shape [N, 2]
            probs = torch.sigmoid(out)[:, 1]

            # Only CPU when appending
            y_scores_m050.append(probs.cpu())
            y_true_m050.append(data.y.cpu())

    y_true_m010 = torch.cat(y_true_m010).numpy()
    y_scores_m010 = torch.cat(y_scores_m010).numpy()

    y_true_m005 = torch.cat(y_true_m005).numpy()
    y_scores_m005 = torch.cat(y_scores_m005).numpy()

    y_true_m100 = torch.cat(y_true_m100).numpy()
    y_scores_m100 = torch.cat(y_scores_m100).numpy()

    y_true_m050 = torch.cat(y_true_m050).numpy()
    y_scores_m050 = torch.cat(y_scores_m050).numpy()

    fpr005, tpr005, thresholds005 = roc_curve(y_true_m005, y_scores_m005)
    roc_auc_005 = auc(fpr005, tpr005)

    fpr050, tpr050, thresholds050 = roc_curve(y_true_m050, y_scores_m050)
    roc_auc_050 = auc(fpr050, tpr050)

    fpr010, tpr010, thresholds010 = roc_curve(y_true_m010, y_scores_m010)
    roc_auc_010 = auc(fpr010, tpr010)

    fpr100, tpr100, thresholds100 = roc_curve(y_true_m100, y_scores_m100)
    roc_auc_100 = auc(fpr100, tpr100)


    


    

    
    







    
