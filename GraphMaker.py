#This script takes a file that has been converted to .npz (so triggering and cuts applied), and turns it into a graph for the GNN. 
#Use like --> python3 GraphMaker.py <Input .npz file> <outpath>

#Imports
import awkward as ak
import numpy as np
import uproot
import torch
from torch_geometric.data import Data, Dataset
import os
import sys

#dataset class
class FileGraphDataset(Dataset):
    def __init__(self, root, file_number=None, signal_status=None, data_list=None, transform=None, pre_transform=None):
        self.file_number = file_number
        self.data_list = data_list
        self.signal_status = int(signal_status)
        super().__init__(root, transform, pre_transform)

        if self.data_list is None: #ie, trying to access 
            self.data_list = torch.load(self.processed_paths[0])
        else: #ie, trying to save
            os.makedirs(self.processed_dir, exist_ok=True)
            torch.save(self.data_list, self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.signal_status == 0:
            self.converted_status = 'background'
        if self.signal_status == 1:
            self.converted_status = 'signal'
        return( [f'batch2_cleaned_file_{self.converted_status}_{self.file_number}_graphs.pt'])

    def len(self):
        return( len(self.data_list) ) #we require datasets to have both a length attribute

    def get(self, idx):
        return(self.data_list[idx]) #as well as a get function to grab by index. 



#Main Function
def main():
    #parse command line arguments
    hcal_hits_file = np.load(sys.argv[1]) #open the .npz file
    hcal_hits_array = hcal_hits_file["hcal_hits_array"] #grab the hits array

    output_file_path = sys.argv[2] #just a path! including ending /, so like /home/wer2ct/file_place/

    total_events = np.unique(hcal_hits_array[:,1])

    #create the graphs
    graph_list = []

    file_number_ = torch.tensor(int(hcal_hits_array[0,0]), dtype = torch.long) #grab the file number (should be the same for all events in the input file ofc)
    signal_status_ = torch.tensor([int(hcal_hits_array[0,2])], dtype = torch.long) #grab the signal status

    print(f"Beginning graph creation for file {file_number_}, signal status = {signal_status_[0]}")
    
    for i in np.unique(hcal_hits_array[:,1]):
        #for debugging / delete before implementation
        
        event_sub_array = hcal_hits_array[hcal_hits_array[:,1] == i]
        
        #Now we want to do some data quality stuff, strike OOV hits, and hits in the side HCal. 
        hcal_hit_x = event_sub_array[:,3] 
        hcal_hit_y = event_sub_array[:,4] 
        hcal_hit_section = event_sub_array[:,-1] 
        oov_x_idx = list(np.where(abs(hcal_hit_x) > 1005)[0]) #out of volume for 2m long bar. 
        oov_y_idx = list(np.where(abs(hcal_hit_y) > 1005)[0])
        hcal_section_idx = list(np.where(hcal_hit_section != 0)[0]) #grab indices for hits in the side-hcal. remove them. 
        #print(oov_x_idx)
        #print(oov_y_idx)
        #print(hcal_section_idx)
        bad_indices = list(set(oov_x_idx + oov_y_idx + hcal_section_idx)) #make sure no duplicate indices.
        cleaned_event_sub_array = np.delete(event_sub_array, bad_indices, axis = 0)
        #Normalize z to make sure that we are only looking a relative distances in z (cuz of signal decay distance bias)
        raw_node_x = cleaned_event_sub_array[:,3] 
        raw_node_y = cleaned_event_sub_array[:,4]
        raw_node_z = cleaned_event_sub_array[:,5]
        graph_hit_layers = cleaned_event_sub_array[:,7]
        norm_node_z = (raw_node_z - min(raw_node_z)) / (max(raw_node_z) - min(raw_node_z))
        node_energy = cleaned_event_sub_array[:,6]
        first_hit_layer = min(graph_hit_layers) #grab the layer of the earliest hit
        node_features = np.column_stack((raw_node_x, raw_node_y, norm_node_z, node_energy))
        feature_vector = torch.tensor((node_features), dtype = torch.float) #these are our nodes.
        event_number = torch.tensor(int(event_sub_array[0,1]), dtype = torch.long)
        event_first_layer = torch.tensor(int(first_hit_layer), dtype = torch.long)
        edges = set() #automatically remove any duplicates (there shouldn't be any but just in case I'm silly)
    
        #create edge list:
        for i in range(len(node_features)):
            for j in range(len(node_features)):
                if (i != j): #no self connected nodes. 
                    edges.add((i,j))
    
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous() #need to transpose since torch expects EdgeIndex like [2, # edges], we produced [# edges, 2] contiguous is a memory flag for gpu opt.
        graph_data = Data( x = feature_vector, edge_index = edge_index, y = signal_status_)
        graph_data.file_number = file_number_
        graph_data.graph_number = event_number
        graph_data.first_layer = event_first_layer
        graph_list.append(graph_data)

    FileGraphDataset(root=output_file_path, data_list = graph_list, file_number=file_number_, signal_status=signal_status_[0]) #this command saves our graphs

    print(f"Graphs Saved to {output_file_path}")

main()
    
    














    
