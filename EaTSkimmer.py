#The role of this script is to take a EaT signal or background file, and convert events that pass the Ecal and Hcal Energy requirements into serialized .npz files. This is a file format that is easier to work with for the purposes of creating graphs and image input.

#Use like --> python3 EaTSkimmer.py <Input Root File> <Is Signal? (1 for true, 0 for false)> <file number> <outfile location>

#Imports
import awkward as ak
import numpy as np
import uproot
import sys

#Globals (may make configurable later)
ecal_energy_threshold = 3160 #MeV
hcal_energy_threshold = 4840 #MeV

#Functions

#Main Function
def main():
    #parse command line arguments
    input_file = sys.argv[1]
    is_signal = bool(int(sys.argv[2]))
    print(f"registered is signal {is_signal}")
    file_number = int(sys.argv[3])
    outfile = sys.argv[4]
    
    #Select proper pass names depending on signal or background
    if is_signal: 
        #for signal
        hcal_rec_pass = 'eat_vis'
        ecal_rec_pass = 'eat_vis'
        is_noise_name = 'is_noise_'
    else:
        #for background
        hcal_rec_pass = "eat"
        ecal_rec_pass = "eat"
        is_noise_name = "isNoise_"

    #Loop over events, evaluating cuts:
    print("Starting Event Processing")
    with uproot.open(input_file) as f:
        big_tree = f["LDMX_Events"]
        total_events = big_tree.num_entries

        #load branches into memory
        branches = {
            "ecal_energy": f"EcalRecHits_{ecal_rec_pass}.energy_",
            "ecal_noise": f"EcalRecHits_{ecal_rec_pass}.{is_noise_name}",
            "hcal_energy": f"HcalRecHits_{hcal_rec_pass}.energy_",
            "hcal_section": f"HcalRecHits_{hcal_rec_pass}.section_",
            "hcal_x": f"HcalRecHits_{hcal_rec_pass}.xpos_",
            "hcal_y": f"HcalRecHits_{hcal_rec_pass}.ypos_",
            "hcal_z": f"HcalRecHits_{hcal_rec_pass}.zpos_",
            "hcal_layer": f"HcalRecHits_{hcal_rec_pass}.layer_",
            "hcal_bar": f"HcalRecHits_{hcal_rec_pass}.strip_",
            "hcal_orient": f"HcalRecHits_{hcal_rec_pass}.orientation_",
            "ecal_z": f"EcalRecHits_{ecal_rec_pass}.zpos_",
        }
        arrays = big_tree.arrays(branches.values(), library="ak")
        #<=541.722 for trigger. 
        # Grab branches important to making cuts. 
        ecal_energy = arrays[branches["ecal_energy"]]
        ecal_z = arrays[branches["ecal_z"]]
        ecal_noise = arrays[branches["ecal_noise"]]
        hcal_energy = arrays[branches["hcal_energy"]]
        hcal_section = arrays[branches["hcal_section"]]
        

        #Make our cuts, using awkward's vectorization (this is new to me! super useful). These apply our cuts across all events at once

        #Trigger 
        trigger_mask = ecal_z <= 541.722
        ecal_trigger_energy = ak.sum(ecal_energy * (trigger_mask), axis=1)
        trigger_pass = ecal_trigger_energy < 3160
        n_trigger_pass = ak.sum(trigger_pass)
        
        #ECal Energy
        ecal_effective_energy = ak.sum(ecal_energy * (~ecal_noise), axis=1) #ecal_energy * ~ecal_noise applies masking
        ecal_pass = ecal_effective_energy < ecal_energy_threshold
        trigger_and_ecal = trigger_pass & ecal_pass
        n_trigger_and_ecal = ak.sum(trigger_and_ecal)
        n_ecal_pass = ak.sum(ecal_pass)

        #HCal Energy
        hcal_mask = hcal_section == 0
        hcal_effective = 12 * ak.sum(hcal_energy * hcal_mask, axis=1) #same deal, we create a mask of the section the multiply to apply it. 
        hcal_pass = hcal_effective > 4840

        #Combined Cut
        event_mask = ecal_pass & hcal_pass #requires both Ecal and Hcal conditions met
        print(f"Total events: {len(event_mask)}")
        print(f"Passing Trigger: {n_trigger_pass}")
        print(f"Passing Trigger and ECal requirement: {n_trigger_and_ecal}")
        print(f"Passing ECal requirement: {n_ecal_pass}")
        print(f"Passing Trigger, ECal, and HCal requirement: {ak.sum(event_mask)}")
        print(f"Efficiency: {(ak.sum(event_mask) / len(event_mask))}")

        #Now apply the mask to all of our branches:
        hcal_x = arrays[branches["hcal_x"]][event_mask]
        hcal_y = arrays[branches["hcal_y"]][event_mask]
        hcal_z = arrays[branches["hcal_z"]][event_mask]
        hcal_layer = arrays[branches["hcal_layer"]][event_mask]
        hcal_bar = arrays[branches["hcal_bar"]][event_mask]
        hcal_orient = arrays[branches["hcal_orient"]][event_mask]
        hcal_energy_pass = 12*arrays[branches["hcal_energy"]][event_mask]

        #Importantly want to preserve which event each hit belongs to!! 
        #We can broadcast the initial local index to the size of one our hit branches and then apply same mask. 
        placeholder_array = arrays[branches["hcal_x"]]
        event_ids = (ak.broadcast_arrays(ak.local_index(placeholder_array, axis=0), placeholder_array))[0]
        passed_ids = event_ids[event_mask]
        file_numbers = ak.broadcast_arrays(file_number, hcal_x)[0]
        signal_status = ak.broadcast_arrays(int(is_signal), hcal_x)[0]
        file_numbers_flat = ak.to_numpy(ak.flatten(file_numbers))
        event_ids_flat = ak.to_numpy(ak.flatten(passed_ids))
        signal_status_flat = ak.to_numpy(ak.flatten(signal_status))
        hcal_x_flat = ak.to_numpy(ak.flatten(hcal_x))
        hcal_y_flat = ak.to_numpy(ak.flatten(hcal_y))
        hcal_z_flat = ak.to_numpy(ak.flatten(hcal_z))
        hcal_energy_flat = ak.to_numpy(ak.flatten(hcal_energy_pass))
        hcal_layer_flat = ak.to_numpy(ak.flatten(hcal_layer))
        hcal_bar_flat = ak.to_numpy(ak.flatten(hcal_bar))
        hcal_orientation_flat = ak.to_numpy(ak.flatten(hcal_orient))

        #Combine into one big array
        # Array contents -> |Event Number*|Hcal x|Hcal y|Hcal z|Hcal layer|Hcal bar|Orientation|Hcal Energy|Signal Status*|File Number*| *indicates event-wise

        #A lot!
        output_array = np.column_stack((file_numbers_flat, event_ids_flat, signal_status_flat, hcal_x_flat, hcal_y_flat, hcal_z_flat , hcal_energy_flat, hcal_layer_flat, hcal_bar_flat, hcal_orientation_flat))

        #make a little statistics array
        total_events = len(event_mask)
        passing_ecal_percent = n_ecal_pass / total_events
        passing_hcal_percent = ak.sum(event_mask) / total_events
        stats = np.array(((total_events), (passing_ecal_percent), (passing_hcal_percent) ))
        print(stats)
        #Now save the array to a .npz file
        
        #make a string for if signal or not
        if int(sys.argv[2]) == 0:
            signal_string = 'background'
        if int(sys.argv[2]) == 1:
            signal_string = 'signal'
        
        np.savez(outfile+f'batch3_filtered_{signal_string}_{file_number}.npz', hcal_hits_array = output_array, stats_array = stats)

    print(f"Processed file, saved to {outfile}")

main()

            
            
        




        
    
    
    
