import os

from _01_load_data import generate_balanced_dataset, load_data
from _04_generate_characteristics import generate_characteristics
from _05_prepare_input import split_data
from _06_find_hyperparameters import search_hyperparameters
from _07_generate_xgb_model import generate_model
from _09_generate_stats import generate_stats
from _10_classify_data import prepare_input, generate_results

"""
The overall pipeline for the alpha estimation in ANDI challenge.

The file contains both the possibility of full run (with data generation, preparation of the model etc.)
and the simple estimation run for the new data, using saved-down 1D and 2D models.
For the latter, please proceed to PREDICTIONS section below
"""

#%% DATA SIMULATION AND MODEL GENERATION
#### Parameter setting ####

### Data preparation
generate_data = True
## If true, generate data
path_for_generation = 'My_datasets/Base/'       # path for data generation
no_paths = 10000                                # number of trajectories to generate 
dimensions = [1,2]                              # dimensions to generate data

## If false, load data
path_to_load = 'development_dataset_for_training'
trajectories_file = "task1.txt"
targets_file = "ref1.txt",
folders = "DevData"

### Model's statistics ###
stats = True                                    # Whether to generate model's statistics

#### Run algorithm ####

### Data generation ###
if generate_data:
    generate_balanced_dataset(N=no_paths, dimensions=dimensions, 
                              save=True, save_path=path_for_generation)
    trajectories_file = "task1.txt"
    targets_file = "ref1.txt"
    path_to_load = path_for_generation
    folders = os.path.basename(os.path.normpath(path_for_generation))

### Data preparation ###
load_data(trajectories_file=trajectories_file, targets_file=targets_file, folders = "Base",
          output_folder="Data", dev_training_folder=path_to_load)

### Calculation of features and model ###
for dim in dimensions:
    sf = folders + "_subtask_" + str(dim) + "D"
    print('Dimension: '+str(dim)+"D")
    
    ## Calculate features ##
    print('Calculate features.')
    generate_characteristics(simulation_folder=sf, dimension=dim)
    ## Prepare model input ##
    print('Prepare model input.')
    split_data(simulation_folder=sf, target_file='alphas.txt')
    ## Find hyperparameters ##
    print('Find hyperparameters.')
    search_hyperparameters(simulation_folder=sf)
    ## Generate model ##
    print('Generate model.')
    generate_model(simulation_folder=sf)
    ## Generate base statistics (optional)
    if stats:
        print('Generate statistics.')
        generate_stats(simulation_folder=sf)
        
        
#%% PREDICTIONS
        
#### Parameter setting ####        
### New data classification ###
path_to_input_pred_data = 'challenge_for_scoring'
trajectories_input_file = 'task1.txt'               # File with trajectories input
estimation_dimensions = [1,2]                       # Dimensions for which run estimation
results_files = ['Results/task1_1.txt',             # Names of files with results, per dimension
                 'Results/task1_2.txt'] 
use_andi_submission_models = True                   # Whether to use the exact models used for ANDI challenge submission

## If use_andi_submission_models is false, use any generated models with paths specified below (per dimension)
## The folders schema needs to be as generated with model generation section above
models_folders = ['Auto_1D_subtask_1D', 'Auto_subtask_2D']

#### Run ####
### Make prediciton ###
for dim in estimation_dimensions:
    print('Prepare the input.')
    prepare_input(dev_training_folder=path_to_input_pred_data, trajectories_file=trajectories_input_file, 
                  dim=dim)
    sf2 = path_to_input_pred_data + "_" + str(dim) + "D"
    rf = results_files[estimation_dimensions.index(dim)]
    
    print('Make prediction.')
    if use_andi_submission_models:
        mf = 'Task1_' + str(dim) + "D"
        generate_results(simulation_folder=sf2, model_folder=mf, dim=dim, 
                         result_file=rf)
    else:
        mf = models_folders[dimensions.index(dim)]
        generate_results(simulation_folder=sf2, model_folder=mf, dim=dim, 
                         result_file=rf)
            