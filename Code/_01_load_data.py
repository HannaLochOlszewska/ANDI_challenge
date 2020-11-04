import os
import numpy as np
import pandas as pd
import andi

def generate_balanced_dataset(N, dimensions, save, save_path):
    
    """
    Simple wrapper for generation of balanced dataset for task 1 in ANDI challenge.
    :param N: int, number of trajectories
    :param dimensions: list of ints, dimensions for which generate datasets
    :param save: bool, whether to save data
    :param save_path: string, the directory for saving data
    :return X1, Y1, X2, Y2, X3, Y3: numpy arrays, X and Y data in 3 dimensions
            (if dimension was not requested, a particular array is empty)
    """
    
    project_directory = os.path.dirname(os.getcwd())
    save_directory = os.path.join(project_directory,save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    AD = andi.andi_datasets()    
    X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N = N, tasks = 1, dimensions = dimensions, 
                                         save_dataset=save, path_datasets = save_directory)

    return X1, Y1, X2, Y2, X3, Y3

def load_data(trajectories_file, targets_file, dev_training_folder, folders,
              output_folder="Data"):
    
    """
    Function for loading the provided/generated dataset and saving it in the proper format
    :param trajectories_file: string, name of the file with trajectories
    :param targets_file: string, name of file with alphas (targets in our regression), or None, if only input provided
    :param dev_training_folder: the name of the folder with trajectories_file and targets_file
    :param folders: string, the prefix for the folders' names
    :param output_folder: string, name of parent directory for folders
    """

    project_directory =  os.path.dirname(os.getcwd())
    path_to_files = os.path.join(project_directory, dev_training_folder)
    path_to_output = os.path.join(project_directory, output_folder)
    
    with open(os.path.join(path_to_files,trajectories_file),'r') as file:
        inside = file.readlines()
        trajectories = [x.strip('\n').split(';') for x in inside]
            
    T1 = os.path.join(path_to_output,folders+'_subtask_1D')
    T1_sub = os.path.join(T1,'Trajectories')
    T2 = os.path.join(path_to_output,folders+'_subtask_2D')
    T2_sub = os.path.join(T2,'Trajectories')
    T3 = os.path.join(path_to_output,folders+'_subtask_3D')
    T3_sub = os.path.join(T3,'Trajectories')
    if not os.path.exists(T1):
        os.makedirs(T1)
    if not os.path.exists(T1_sub):
        os.makedirs(T1_sub)
    if not os.path.exists(T2):
        os.makedirs(T2)
    if not os.path.exists(T2_sub):
        os.makedirs(T2_sub)
    if not os.path.exists(T3):
        os.makedirs(T3)
    if not os.path.exists(T3_sub):
        os.makedirs(T3_sub)
        
    if targets_file is not None:
        with open(os.path.join(path_to_files,targets_file),'r') as tfile:
            res_to_fit = tfile.readlines()
        alphas = [x.strip('\n').split(';') for x in res_to_fit]
                
        count_1D, count_2D, count_3D = [0,0,0]
        alphas_1D, alphas_2D, alphas_3D = {}, {}, {}
        for i in range(len(trajectories)):
            x = trajectories[i]
            alpha = alphas[i][1]
            if x[0]=='1.0':
                tr = pd.DataFrame({'x': np.asarray(x[1:],dtype=float)})
                tr.to_csv(os.path.join(T1_sub,str(count_1D)+'_1D.txt'))
                count_1D += 1
                alphas_1D[count_1D] = alpha
            if x[0]=='2.0':
                reshaped = np.asarray(x[1:], dtype=float).reshape(2,int(len(x[1:])/2))
                tr = pd.DataFrame({'x': reshaped[0], 'y': reshaped[1]})
                tr.to_csv(os.path.join(path_to_output,folders+'_subtask_2D','Trajectories',str(count_2D)+'_2D.txt'))
                count_2D += 1
                alphas_2D[count_2D] = alpha
            if x[0]=='3.0':
                reshaped = np.asarray(x[1:], dtype=float).reshape(3,int(len(x[1:])/3))
                tr = pd.DataFrame({'x': reshaped[0], 'y': reshaped[1], 'z': reshaped[2]})
                tr.to_csv(os.path.join(path_to_output,folders+'_subtask_3D','Trajectories',str(count_3D)+'_3D.txt'))
                count_3D += 1
                alphas_3D[count_3D] = alpha
                
        a_1D = pd.DataFrame.from_dict(alphas_1D, orient='index', columns=['Alpha'])
        a_1D.to_csv(os.path.join(path_to_output,folders+'_subtask_1D','alphas.txt'), index=False)
        
        a_2D = pd.DataFrame.from_dict(alphas_2D, orient='index', columns=['Alpha'])
        a_2D.to_csv(os.path.join(path_to_output,folders+'_subtask_2D','alphas.txt'), index=False)
        
        a_3D = pd.DataFrame.from_dict(alphas_3D, orient='index',)
        a_3D.to_csv(os.path.join(path_to_output,folders+'_subtask_3D','alphas.txt'), index=False)
        
    else:
        count_1D, count_2D, count_3D = [0,0,0]
        for i in range(len(trajectories)):
            x = trajectories[i]
            if x[0]=='1.0':
                tr = pd.DataFrame({'x': np.asarray(x[1:],dtype=float)})
                tr.to_csv(os.path.join(T1_sub,str(count_1D)+'_1D.txt'))
                count_1D += 1
            if x[0]=='2.0':
                reshaped = np.asarray(x[1:], dtype=float).reshape(2,int(len(x[1:])/2))
                tr = pd.DataFrame({'x': reshaped[0], 'y': reshaped[1]})
                tr.to_csv(os.path.join(T2_sub,str(count_2D)+'_2D.txt'))
                count_2D += 1
            if x[0]=='3.0':
                reshaped = np.asarray(x[1:], dtype=float).reshape(3,int(len(x[1:])/3))
                tr = pd.DataFrame({'x': reshaped[0], 'y': reshaped[1], 'z': reshaped[2]})
                tr.to_csv(os.path.join(T3_sub,str(count_3D)+'_3D.txt'))
                count_3D += 1
    

if __name__ == "__main__":
    
    generate_balanced_dataset(N=10000, tasks=[1], dimensions=[1,2], save=True, save_path='../My_datasets/Base/')
    
    load_data(trajectories_file="task1.txt", targets_file="ref1.txt", folders = "Base",
              output_folder="Data", dev_training_folder='My_datasets\Base')
    