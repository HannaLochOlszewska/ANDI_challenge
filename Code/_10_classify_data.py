import os

import joblib
import numpy as np
import pandas as pd

import multiprocessing as mp
from itertools import repeat
                      
from _03_characteristics import CharacteristicFour

def get_characteristics(trajectory, dim):
    """
    Function return characteristics for given scenario
    :param char_set: enum, information about characteristic set scenario
    :param path_to_file: str, path to file with trajectory
    :param typ: str, type of diffusion i.e sub, super, rand
    :param motion: str, mode of diffusion eg. normal, directed
    """
    
    if dim == 1:
        x = trajectory['x'].values
        y = None
        z = None
    elif dim == 2:
        x = trajectory['x'].values
        y = trajectory['y'].values
        z = None    
    elif dim == 3:
        raise ValueError("Only 1D and 2D are available.")
        
    ch = CharacteristicFour(x=x, y=y, z=z, dim=dim, file="", percentage_max_n=0.1, typ="", motion="")
    data = ch.data
    
    return data

def prepare_input(dev_training_folder, trajectories_file, dim):
    """
    Function for loading the provided data for prediction and saving it in the proper format to plug into model
    :param dev_training_folder: the name of the folder with trajectories_file and targets_file
    :param trajectories_file: string, name of the file with trajectories
    :param dim: int, dimension
    :return: none, the characteristic file and npy file with data saved as side effect
    """
    
    project_directory = os.path.dirname(os.getcwd())
    path_to_files = os.path.join(project_directory, dev_training_folder)
    
    with open(os.path.join(path_to_files,trajectories_file),'r') as file:
        inside = file.readlines()
        trajectories = [x.strip('\n').split(';') for x in inside]
        
    path_to_data = os.path.join(project_directory, "Data", dev_training_folder + "_" + str(dim) + "D")    
    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
    path_to_characteristics_data = os.path.join(path_to_data, "Characteristics")
    if not os.path.exists(path_to_characteristics_data):
        os.makedirs(path_to_characteristics_data)
            
    trs = []
    
    if dim == 1:
        for i in range(len(trajectories)):
            x = trajectories[i]
            if int(float(x[0])) == 1:
                reshaped = np.asarray(x[1:], dtype=float).reshape(dim,int(len(x[1:])/dim))
                tr = pd.DataFrame({'x': reshaped[0]})
                trs.append(tr)
    elif dim == 2:
        for i in range(len(trajectories)):
            x = trajectories[i]
            if int(float(x[0])) == 2:
                reshaped = np.asarray(x[1:], dtype=float).reshape(dim,int(len(x[1:])/dim))
                tr = pd.DataFrame({'x': reshaped[0], 'y': reshaped[1]})
                trs.append(tr)
                
    #trs=trs[12972:] # 377, 750, 12972?
    #trs = trs[2543:2545]
                
    characteristics_input = zip(trs, repeat(dim))
    pool = mp.Pool(processes=(mp.cpu_count() - 1))
    characteristics_data = pool.starmap(get_characteristics, characteristics_input)
    pool.close()
    pool.join()
    characteristics_data = pd.concat(characteristics_data)

    characteristics_data.to_csv(os.path.join(path_to_characteristics_data, "characteristics.csv"), index=False)
    
    characteristics_data = characteristics_data.drop(["file", "Alpha", "motion"], axis=1)
    
    X = characteristics_data.values
    np.save(os.path.join(path_to_characteristics_data, "X_data.npy"), X)
    
    return


def generate_results(simulation_folder, model_folder, dim, result_file):
    """
    Function for generating statistics about model
    :param simulation_folder: str, name of subfolder for given data set
    :param model_folder: str, the name of folder with model to apply
    :param dim: int, dimension
    :param result_file: str, path to file to store results
    :return: none, predictions saved as side effect
    """

    project_directory = os.path.dirname(os.getcwd())
    path_to_data = os.path.join(project_directory, "Data", simulation_folder)
    path_to_characteristics_data = os.path.join(path_to_data, "Characteristics")
    path_to_scenario = os.path.join(project_directory, "Models", model_folder,
                                    "XGB", "Model")
    path_to_results = os.path.join(project_directory, os.path.basename(result_file))
    if not os.path.exists(os.path.dirname(path_to_results)):
        os.makedirs(os.path.dirname(path_to_results))
                
    path_to_model = os.path.join(path_to_scenario, "model.sav")
    model = joblib.load(path_to_model)

    X = np.load(os.path.join(path_to_characteristics_data,"X_data.npy"), allow_pickle=True)
    y_pred = model.predict(X)
    
    with open(os.path.join(project_directory,result_file),'w') as file:
        for y in y_pred:
            file.write(str(dim)+';'+str(y)+'\n')
            
    return


if __name__ == "__main__":
    
    prepare_input(dev_training_folder='challenge_for_scoring', trajectories_file='task2.txt', dim=1)
    
    generate_results(simulation_folder="challenge_for_scoring_1D", model_folder='T2_subtask_1D',
                     dim=1, result_file="Results/task2.txt")
