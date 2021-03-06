# ANDI_challenge

### Hanna Loch-Olszewska

4.11.2020

This repository contains Python code for the anomalous diffusion exponent estimation using the machine learning feature-based method - gradient boosting. It has been used for the contribution to ANDI challenge (see [https://competitions.codalab.org/competitions/23601](https://competitions.codalab.org/competitions/23601) and [https://doi.org/10.5281/zenodo.3707702](https://doi.org/10.5281/zenodo.3707702)).

## Algorithm

The algorithm is based on the ML gradient boosting method. As the input to regressor, we use the following features:  
 * anomalous diffusion exponent alpha [2],
 * diffusion constant D [2],
 * fractal dimension [3],
 * Gaussianity [3],
 * straightness [3],
 * MSD ratio [3],
 * power fitted to p-variation for lags 1-5 [2],
 * alpha estimate including noise - 3 methods according to [4],
 * velocity autocorrelation function for lag 1 [6],
 * normalised maximum excursion [6],
 * p-variation based statistics [6].
 
We use the optimal gradient boosting implementation from `xgboost` Python library.

## Requirements
1. `andi_datasets` for data generation.
2. `numpy`, `pandas` and `scipy` for data handling and features calculation.
3. `multiprocessing` for multi-thread calculations. 
4. `sklearn` for ML utilities.
5. `xgboost` for the gradient boosting algorithms.
6. `joblib` and `json` for model and hyperparameters storage, respectively.

## Usage

The python file *run_algorithm.py* contains schema and the pipeline for data analysis. It is assumed that the given data are supplied in the format as for the ANDI challenge (see Ref. [1]).  
In order to properly run the script, the dataset (for example the challenge one) should be placed in the same directory where the repository is stored.

## References
[1] G. Muñoz-Gil, G. Volpe, M. A. García-March, R. Metzler, M. Lewenstein, C. Manzo (2020), "The anomalous diffusion challenge: single trajectory characterisation as a competition". Proc. SPIE 11469, Emerging Topics in Artificial Intelligence 2020, 114691C, doi: [10.1117/12.2567914](https://doi.org/10.1117/12.2567914).

[2] J. Janczura, P. Kowalek, H. Loch-Olszewska, J. Szwabiński, A. Weron (2020), "Classification of particle trajectories in living cells: machine learning versus statistical testing hypothesis for fractional anomalous diffusion", Physical Review E 102, 032402, doi: [10.1103/PhysRevE.102.032402](https://doi.org/10.1103/PhysRevE.102.032402). 

[3] P. Kowalek, H. Loch-Olszewska, J. Szwabiński (2019), "Classification of diffusion modes in single-particle tracking data: Feature-based versus deep-learning approach". Physical Review E 100, 032410, doi: [10.1103/PhysRevE.100.032410](https://doi.org/10.1103/PhysRevE.100.032410). 

[4] G. Sikora, M. Teuerle, A. Wyłomańska, and D. Grebenkov (2017), "Statistical properties of the anomalous scaling exponent estimator based on time-averaged mean-square displacement". Phys. Rev. E 96, 022132, doi: [10.1103/PhysRevE.96.022132](https://doi.org/10.1103/PhysRevE.96.022132).

[5] Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska (2018), "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data". Phys. Rev. E 98, 062139, doi: [10.1103/PhysRevE.98.062139](https://doi.org/10.1103/PhysRevE.98.062139).

[6] H. Loch-Olszewska, J. Szwabiński (2020) "Impact of feature choice on machine learning classification of fractional anomalous diffusion", manuscript in preparation.