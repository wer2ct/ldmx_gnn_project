#This script trains our GNN model 

#Use like --> python3 GNN_model

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

#Globals
SEED = 2026
np.random.seed(SEED)    # Setting the seed for reproducibility
torch.manual_seed(SEED)

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


#Training and Validation Loop
def run_train(model, training_loader_, validation_loader_, num_iterations=100, 
              log_dir='/scratch/wer2ct/December2025/', 
              log_prefix='bigger_run_GNN_v2', optimizer='Adam', lr=0.001, max_epochs_ = 15):

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} ({num_gpus} GPUs available)")

    # Move model to device (after wrapping, or just alone if it hasn't been wrapped)
    model = model.to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_fn = getattr(torch.optim, optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=lr)

    iteration = 0
    epoch_at_valid = []
    validation_acc = []

    #adding a step to make the log. directory if it does not exist. 
    os.makedirs(log_dir, exist_ok=True)
    train_log_name = f'{log_dir}/{log_prefix}train.csv'
    val_log_name = f'{log_dir}/{log_prefix}val.csv'
    
    with open(train_log_name, 'w', newline='') as trainfile, \
         open(val_log_name, 'w', newline='') as valfile:
        train_writer = csv.writer(trainfile)
        val_writer = csv.writer(valfile)
        train_writer.writerow(['iter', 'epoch', 'loss'])
        val_writer.writerow(['iter', 'epoch', 'loss', 'accuracy'])

        max_epochs = max_epochs_

        epoch = 0
        while epoch < max_epochs:
            #placing into training mode
            model.train()
            epoch += 1 
            for training_data in training_loader_: #this is looping over the batches
                training_data = training_data.to(device) #put data onto our gpu / cpu
                # Training step
                optimizer.zero_grad()
                #let the model do its prediction --> this is basically the input to the forward method. 
                model_out = model(training_data.x , training_data.edge_index , training_data.batch)
                #out loss here is cross entropy, performs a softmax on the class vector, then computes the loss. 
                loss = criterion(model_out, training_data.y)
                #backpropagation -> using Adam
                loss.backward()
                optimizer.step()
                #write our output. 
                train_writer.writerow([iteration, epoch, loss.item()])
                iteration += 1
                if iteration % 100 == 0:
                    print(iteration)

            # Do a validation every epoch (its just easier)
            model.eval()
            print(f"completed training on epoch {epoch}")
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                total_loss = 0.0
                for validation_data in validation_loader_:
                    validation_data = validation_data.to(device)
                    out = model(validation_data.x, validation_data.edge_index, validation_data.batch)
                    val_loss = criterion(out, validation_data.y)
                    total_loss += val_loss.item()
                    #next check for the # of correct classifications, check argmax of our logits, 
                    n_correct = torch.sum(out.argmax(dim=1) == validation_data.y)
                    total_correct += n_correct.item()
                    total_samples += len(validation_data.y)
                
                acc = total_correct / total_samples
                avg_loss = total_loss / len(validation_loader_) 
                print(f"Validation at epoch {epoch}, "
                              f"accuracy = {acc:.4f}, loss = {avg_loss:.4f}")
                
                val_writer.writerow([iteration, epoch, avg_loss, acc]) 

#main function
def main():
    #This thing is basically not going to be configurable at all from the command line for now. 
    #Need to load in all the necessary training and validation files

    #Grab the file paths 
    training_directory = "/scratch/wer2ct/December2025/final_runs/training/"
    validation_directory = "/scratch/wer2ct/December2025/final_runs/roc_validation/"
    signal_training_paths = glob.glob(os.path.join(training_directory, "m*"))
    bkgs_training_paths = glob.glob(os.path.join(training_directory, "b*"))
    signal_validation_paths = glob.glob(os.path.join(validation_directory, "m*"))
    bkgs_validation_paths = glob.glob(os.path.join(validation_directory, "b*"))

    #Load the signal training files
    print("Beginning to load training files")
    signal_training_file_list = []
    for path in signal_training_paths:
        signal_training_file_list.append((torch.load(path, weights_only = False)))
        print(f"Loaded: {path}")

    #Load the background training files
    training_background = torch.load(bkgs_training_paths[0], weights_only = False)
    print("Loaded all training files")
    
    #Concatenating our training files, creating a training loader
    full_dataset_list_training = training_background + signal_training_file_list
    full_dataset_training = ConcatDataset(full_dataset_list_training)
    training_loader = DataLoader(full_dataset_training, batch_size = 500, drop_last = True, shuffle=True, num_workers=1)
    print(f'Training on {len(training_loader)*training_loader.batch_size} total graphs')

    #Now loading the signal validation files (for ROC and validation)
    print("Beginning to load validation files")
    signal_val_file_list = []
    for path in signal_validation_paths:
        signal_val_file_list.append((torch.load(path, weights_only = False)))
        print(f"Loaded: {path}")

    #Now loading the background validation files
    val_background = torch.load(bkgs_validation_paths[0], weights_only = False)

    #Concatenating our validation files, creating a validation loader
    full_dataset_list_val = val_background + signal_val_file_list
    full_dataset_val = ConcatDataset(full_dataset_list_val)
    validation_loader =  DataLoader(full_dataset_val, batch_size = 500, drop_last = True, shuffle=True, num_workers=1)
    print(f'Validating on {len(validation_loader)*validation_loader.batch_size} total graphs')

    #Declare our model and begin training!
    print("Beginning to train")
    classifier = GNN_v2(in_channels = 4, hc1 = 20, hc2 = 40, hc3 = 50, fc1 = 25, fc2 = 12, fc3 = 6, out_channels = 2)
    run_train(classifier, training_loader, validation_loader, num_iterations = 1000, optimizer='Adam', log_dir = '/scratch/wer2ct/December2025/final_runs/GNN_output/', log_prefix= 'GNN_v2_final_run', max_epochs_ = 18)

    #Save the output
    print("Finished training, saving weights")
    model_weights = classifier.state_dict()
    torch.save(model_weights, '/scratch/wer2ct/December2025/final_runs/GNN_output/' + 'GNN_v2_final_run_weights.pth')

    #Now that we are done with training, let's go ahead and save some relevant plots
    #First, the training curves:
    train_log = pd.read_csv('/scratch/wer2ct/December2025/final_runs/GNN_output/GNN_v2_final_runtrain.csv')
    val_log = pd.read_csv('/scratch/wer2ct/December2025/final_runs/GNN_output/GNN_v2_final_runval.csv')
    window_size = 3  # adjust this for more or less smoothing
    train_log['loss_smooth'] = train_log['loss'].rolling(window=window_size, min_periods=1).mean()

    # Plot smoothed training loss and validation loss
    plt.figure(figsize=(8,6))
    plt.plot(train_log['iter'], train_log['loss_smooth'], label=f'Training Loss (rolling mean, window={window_size})', alpha=0.8)
    plt.plot(val_log['iter'], val_log['loss'], label='Validation Loss', linewidth=2, color='orange')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Smoothed)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/scratch/wer2ct/December2025/final_runs/GNN_output/training_curve.png")
    plt.close()

    #Now, the validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(val_log['epoch'], val_log['accuracy'], marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/scratch/wer2ct/December2025/final_runs/GNN_output/validation_accuracy.png")
    plt.close()

    print("tasks completed, training curves saved")

main()























