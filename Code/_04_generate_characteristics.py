import multiprocessing as mp
import os
from itertools import repeat
import pandas as pd

from _03_characteristics import CharacteristicFour

"""
Characteristics generators.
The multiprocessing is enabled.
"""

def get_characteristics(path_to_file, fname, dim, typ="", motion=""):
    """
    Function return characteristics for given scenario
    :param path_to_file: string, path to file with trajectory
    :param fname: string, information about characteristic set scenario
    :param dim: int, dimension of trajectory
    :param typ: str, mode of diffusion i.e sub, super, normal
    :param motion: str, model of motion eg. Bm, fBm, ATTM
    """
    print(path_to_file)
    trajectory = pd.read_csv(path_to_file)
    
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
                
    ch = CharacteristicFour(x=x, y=y, z=z, dim=dim, file=fname, percentage_max_n=0.1, typ=typ, motion=motion)
    data = ch.data
    
    return data

def get_characteristics_single(trajectory, path_to_files, dimension, typ="", motion=""):
    """
    :param trajectory: str, trajectory name    
    :param path_to_files: string, path to folder with trajectories
    :param dimension: int, dimension of trajectory
    :param typ: str, mode of diffusion i.e sub, super, normal
    :param motion: str, model of motion eg. Bm, fBm, ATTM
    :return: dataframe with characteristics for single trajectory
    """

    direct_file = os.path.join(path_to_files,trajectory)
    ## HACK: for the data generated with labels, we should pass type and motion here
    d = get_characteristics(direct_file, trajectory, dim=dimension, typ=typ, motion=motion)
    return d

def generate_characteristics(simulation_folder, dimension):
    """
    Function for generating the characteristics file for given scenario
    - characteristics are needed for featured based classifiers
    Function use multiprocessing to speed generating of the characteristics file
    :param simulation_folder: str, name of subfolder for given set simulation
    :param dimension: int, dimension of trajectory
    :return: none, dataframe with characteristics saved as side effect
    """

    project_directory = os.path.dirname(os.getcwd())
    path_to_save = os.path.join(project_directory, "Data", simulation_folder)

    path_to_trajectories = os.path.join(path_to_save, "Trajectories")
    path_to_characteristics_data = os.path.join(path_to_save, "Characteristics")
    if not os.path.exists(path_to_characteristics_data):
        os.makedirs(path_to_characteristics_data)
    trajectories_lists = [file for file in os.listdir(path_to_trajectories)]
    
    characteristics_input = zip(trajectories_lists, repeat(path_to_trajectories), repeat(dimension))
    pool = mp.Pool(processes=(mp.cpu_count() - 1))
    characteristics_data = pool.starmap(get_characteristics_single, characteristics_input)
    pool.close()
    pool.join()
    results = pd.concat(characteristics_data)
    results.to_csv(os.path.join(path_to_characteristics_data, "characteristics.csv"), index=False)

if __name__ == "__main__":

    generate_characteristics(simulation_folder="Base_subtask_1D", dimension=1)