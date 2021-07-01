import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from scipy import linalg as LA
from scipy.stats import kstest, chi2, normaltest
from exponents import FindExponents

from _02_msd import generate_theoretical_msd_normal, generate_empirical_msd, \
                    generate_theoretical_msd_anomalous_log, generate_empirical_pvariation, \
                    generate_empirical_velocity_autocorrelation, \
                    generate_theoretical_msd_anomalous_with_noise, \
                    generate_detrended_moving_average
                    
"""
Classes of characteristics sets.
"""

class CharacteristicBase:
    """
    Class representing base characteristics of given trajectory, based on T. Wagner et al.
    "Classification and Segmentation of Nanoparticle Diffusion Trajectories in Cellular Micro Environments"
    PLoS ONE 12(1), (2017).    
    """

    def __init__(self, x, y, z, dim, file, percentage_max_n, typ="", motion=""):
        """
        :param x: list, x coordinates
        :param y: list, y coordinates
        :param y: list, z coordinates
        :param dim: int, dimension
        :param file: str, path to trajectory
        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
        :param typ: str, type of diffusion i.e sub, super, rand
        :param motion: str, mode of diffusion eg. normal, directed
        """
        self.x = x
        self.y = y
        self.z = z
        self.percentage_max_n = percentage_max_n
        self.type = typ
        self.motion = motion
        self.file = file
        self.dim = dim

        self.N = self.get_length_of_trajectory()
        self.T = self.get_duration_of_trajectory()
        self.dt = (self.N-1)/self.T
        self.max_number_of_points_in_msd = self.get_max_number_of_points_in_msd()
        self.n_list = self.get_range_for_msd()
        self.empirical_msd = generate_empirical_msd(self.x, self.y, self.n_list)
        self.displacements = self.get_displacements()
        self.d = self.get_max_displacement()
        self.L = self.get_total_length_of_path()

    def get_length_of_trajectory(self):
        """
        :return: int, length of trajectory represented by N parameter
        """
        return len(self.x)

    def get_duration_of_trajectory(self):
        """
        :return: int, duration of the trajectory life represented by T parameter
        """
        ## HACK: for this time being we set T=N
        return self.N-1

    def get_max_number_of_points_in_msd(self):
        """
        :return: int, maximal number which can be used to generate msd
        """
        if self.percentage_max_n != None:
            return max(int(np.floor(self.percentage_max_n * self.N)),4)
        else:
            return self.N if self.N <= 100 else 101

    def get_range_for_msd(self):
        """
        :return: array, range of steps in msd function
        """
        return np.array(range(1, self.max_number_of_points_in_msd))

    def get_displacements(self):
        """
        :return: array, list of displacements between x and y coordinates
        """
        
        if self.dim == 1:
            dips = np.array([self.get_displacement(self.x[i],0,self.x[i - 1],0) for i in range(1, self.N - 1)])
        elif self.dim == 2:
            dips = np.array(
            [self.get_displacement(self.x[i], self.y[i], self.x[i - 1], self.y[i - 1]) for i in range(1, self.N - 1)])
        elif self.dim == 3:
            pass
        
        return dips

    @staticmethod
    def get_displacement(x1, y1, x2, y2):
        """
        :param x1: float, first x coordinate
        :param y1: float, first y coordinate
        :param x2: float, second x coordinate
        :param y2: float, second y coordinate
        :return: float, displacement between two points
        """
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_total_length_of_path(self):
        """
        :return: int, total length of path represented by L parameter
        """
        return sum(self.displacements)

    def get_max_displacement(self):
        """
        :return: float, maximum displacement represented by d in all displacement list
        """
        return max(self.displacements)


class Characteristic(CharacteristicBase):
    """
    Class representing characteristics of given trajectory
    """

    def __init__(self, x, y, z, dim, file, percentage_max_n=1, typ="", motion=""):
        """
        :param x: list, x coordinates
        :param y: list, y coordinates
        :param y: list, z coordinates
        :param dim: int, dimension
        :param file: str, path to trajectory
        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
        :param typ: str, type of diffusion i.e sub, super, rand
        :param motion: str, mode of diffusion eg. normal, directed
        """

        CharacteristicBase.__init__(self,x, y, z, dim, file, percentage_max_n, typ, motion)

        self.D = self.get_diffusion_coef()
        self.alpha = self.get_exponent_alpha()
        self.fractal_dimension = self.get_fractal_dimension()
        self.gaussianity = self.get_gaussianity()
        self.mean_gaussianity = self.get_mean_gaussianity()
        self.mean_squared_displacement_ratio = self.get_mean_squared_displacement_ratio()
        self.straightness = self.get_straightness()

    def get_diffusion_coef(self):
        """
        :return: float, diffusion coefficient represented by D parameter;
        estimated based on curve fitting of empirical and normal theoretical diffusion.
        """
        popt, _ = curve_fit(lambda x, d: generate_theoretical_msd_normal(x, d, self.dt, self.dim), self.n_list,
                            self.empirical_msd)
        D = popt[0]
        return D

    def get_exponent_alpha(self):
        """
        :return: float, exponential anomalous parameter by alpha parameter;
        estimated based on curve fitting of empirical and normal anomalous diffusion.
        Modification of this function can also estimate D parameter
        """
        try:
            popt, _ = curve_fit(
                lambda x, log_D, a: generate_theoretical_msd_anomalous_log(np.log(self.dt * self.n_list), log_D, a, self.dim),
                np.log(self.dt * self.n_list), np.log(self.empirical_msd), bounds=((-np.inf, 0), (np.inf, 2)))
            alpha = popt[1]
        except ValueError:
            alpha = 0
        return alpha

    def get_fractal_dimension(self):
        """
        The fractal dimension is a measure of the space-filling capacity of a pattern.
        :return: float, fractional dimension parameter
        """
        upper = np.log(self.N)
        lower = np.log(self.N * self.L ** (-1) * self.d)
        D = upper / lower
        return D

    def get_gaussianity(self):
        """
        A trajectory’s Gaussianity checks the Gaussian statistics on increments
        :return: array, list of gaussianity points
        """        
        if self.dim == 1:
            r4 = generate_empirical_msd(self.x, np.zeros(len(self.x)), self.n_list, 4)
            r2 = generate_empirical_msd(self.x, np.zeros(len(self.x)), self.n_list, 2)
        elif self.dim == 2:
            r4 = generate_empirical_msd(self.x, self.y, self.n_list, 4)
            r2 = generate_empirical_msd(self.x, self.y, self.n_list, 2)
        elif self.dim == 3:
            pass
        g = r4 / (2 * r2 ** 2)
        g = -1 + 2 * r4 / (3 * r2 ** 2)
        return g

    def get_mean_gaussianity(self):
        """
        :return: float, mean of gaussianity points
        """
        return np.mean(self.gaussianity)

    def get_mean_squared_displacement_ratio(self):
        """
        The mean square displacement ratio characterizes the shape of the MSD curve.
        :return: float, mean squared displacement ratio parameter
        """
        n1 = np.array(range(1, self.max_number_of_points_in_msd - 1))
        n2 = np.array(range(2, self.max_number_of_points_in_msd))
        r_n1 = self.empirical_msd[0:self.max_number_of_points_in_msd - 2]
        r_n2 = self.empirical_msd[1:self.max_number_of_points_in_msd]
        r = np.mean(r_n1 / r_n2 - n1 / n2)
        return r

    def get_straightness(self):
        """
        Straightness is a measure of the average direction change between subsequent steps.
        :return: float, straing
        """
        if self.dim == 1:
            upper = self.get_displacement(self.x[self.N - 2], 0, self.x[0], 0)
            displacements = np.array(
                [self.get_displacement(self.x[i], 0, self.x[i - 1], 0) for i in range(1, self.N - 1)])
        elif self.dim == 2:
            upper = self.get_displacement(self.x[self.N - 2], self.y[self.N - 2], self.x[0], self.y[0])
            displacements = np.array(
                [self.get_displacement(self.x[i], self.y[i], self.x[i - 1], self.y[i - 1]) for i in range(1, self.N - 1)])
        elif self.dim == 3:
            pass
        
        lower = sum(displacements)
        S = upper / lower
        return S


class CharacteristicFour(Characteristic):
    """
    Characteristics for the alpha regression
    """

    def __init__(self, x, y, z, dim, file, percentage_max_n=0.1, typ="", motion=""):
        """
        :param x: list, x coordinates
        :param y: list, y coordinates
        :param y: list, z coordinates
        :param dim: int, dimension
        :param file: str, path to trajectory
        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
        :param typ: str, type of diffusion i.e sub, super, rand
        :param motion: str, mode of diffusion eg. normal, directed
        """

        Characteristic.__init__(self, x, y, z, dim, file, percentage_max_n, typ, motion)
        
#        self.D_new = self.estimate_diffusion_coef()
        self.p_variations, self.p_variation_names = self.get_pvariation_test(p_list=np.arange(1, 6))
        self.velocity_autocorrelation, self.velocity_autocorrelation_names = self.get_velocity_autocorrelation([1])
        self.p_variation = self.get_feature_from_pvariation()
        self.max_excursion_normalised = self.get_max_excursion()
        self.alpha_n_1 = self.estimate_diffusion_exponent_with_noise_1()
        self.alpha_n_2, _ = self.estimate_diffusion_exponent_with_noise_2()
        self.alpha_n_3, _ = self.estimate_diffusion_exponent_with_noise_3()

        self.values = [self.file, self.type, self.motion, self.D, self.alpha,
                       self.alpha_n_1, self.alpha_n_2, self.alpha_n_3, 
                       self.fractal_dimension, self.mean_gaussianity,
                       self.mean_squared_displacement_ratio, self.straightness,
                       self.p_variation, self.max_excursion_normalised] + list(self.velocity_autocorrelation) \
                       + list(self.p_variations)
        self.columns = ["file", "Alpha", "motion", "D", "alpha",
                        "alpha_n_1", "alpha_n_2", "alpha_n_3", 
                        "fractal_dimension", "mean_gaussianity",
                        "mean_squared_displacement_ratio", "straightness",
                        "p-variation", "max_excursion_normalised"] + self.velocity_autocorrelation_names \
                        + self.p_variation_names
                        
        self.data = pd.DataFrame([self.values], columns=self.columns)
        
#    def estimate_diffusion_coef(self):
#        """
#        :return: float, exponential anomalous parameter by alpha parameter;
#        estimated based on curve fitting to log of empirical and normal anomalous diffusion.
#        Modification of this function can also estimate alpha parameter
#        """
#        log_msd = np.log(self.empirical_msd)
#        tau = np.log((self.dt * self.n_list)).reshape((-1, 1))
#        model = LinearRegression().fit(tau, log_msd)
#        log_d = model.intercept_
#        D = math.exp(log_d) / 4
#        return D


    def get_pvariation_test(self, p_list):
        """
        :param p_list: list, p-values for calculation of p-variation
        :return: tuple with list of values and list of strings,
        the list of powers fitted to the calculated p-variations
        and list of the corresponding feature names
        """
        max_m = int(max(0.01 * self.N, 5))
        m_list = np.arange(1, max_m + 1)

        test_values = []
        if self.dim == 1:            
            p_var = generate_empirical_pvariation(self.x, np.zeros(len(self.x)), p_list, m_list)
        elif self.dim == 2:
            p_var = generate_empirical_pvariation(self.x, self.y, p_list, m_list)
        for i in range(len(p_list)):
            pv = p_var[i]
            try:                                    
                gamma_power_fit = LinearRegression().fit(np.log(m_list).reshape(-1, 1), np.log(pv))
                gamma = gamma_power_fit.coef_[0]
            except ValueError:
                gamma = -9999                                                       
            test_values.append(gamma)

        feature_names = ['p_var_' + str(p) for p in p_list]

        return test_values, feature_names
        
    def get_velocity_autocorrelation(self, hc_lag_list):
        """
        Calculate the velocity autocorrelation
        :return: float, the empirical autocorrelation for lag 1.
        """
        # hc_lag_list = [1,2,3,4,5]
        titles = ["vac_lag_" + str(x) for x in hc_lag_list]
        if self.dim == 1:
            autocorr = generate_empirical_velocity_autocorrelation(self.x, np.zeros(len(self.x)), hc_lag_list, self.dt, delta=1)
        elif self.dim == 2:
            autocorr = generate_empirical_velocity_autocorrelation(self.x, self.y, hc_lag_list, self.dt, delta=1)
        return autocorr, titles

    def get_feature_from_pvariation(self):
        """
        Calculate p_variation with preset p and m choice and return info about the 
        :return: int, 
        """
        p_list = [1/H for H in np.arange(0.1, 1.0, 0.1)]
        m_list = list(range(1, 11))
        
        if self.dim == 1:
            p_var_matrix = generate_empirical_pvariation(self.x, np.zeros(len(self.x)), p_list, m_list)            
        elif self.dim == 2:
            p_var_matrix = generate_empirical_pvariation(self.x, self.y, p_list, m_list)

        m_array = np.array(m_list).reshape(-1, 1)
        p_var_d = [LinearRegression().fit(m_array, p_var_matrix[p_index]).coef_[0] for p_index in range(len(p_list))]
        signs_p = np.nonzero(np.diff([np.sign(val) for val in p_var_d]))

        if len(signs_p[0]) > 0:
            p_var_info = signs_p[0][0] * np.sign(p_var_d[0])
        else:
            p_var_info = 0

        return p_var_info

    def get_total_displacement(self):
        """
        The total displacement of the trajectory
        :return: float, the total displacement of a trajectory
        """
        if self.dim == 1:
            total_displacement = self.get_displacement(self.x[self.N - 1], 0, self.x[0], 0)
        elif self.dim == 2:
            total_displacement = self.get_displacement(self.x[self.N - 1], self.y[self.N - 1], self.x[0], self.y[0])
        return total_displacement

    def get_max_excursion(self):
        """
        The maximal excursion of the particle, normalised to its total displacement (range of movement)
        :return: float, max excursion
        """
        try:
            excursion = self.d / self.get_total_displacement()
        except ValueError:
            excursion = 0
        return excursion
        
        
    def estimate_diffusion_exponent_with_noise_1(self):
        """
        The estimation of diffusion exponent with noise, according to method I from
        Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska,
        "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data"
        Phys. Rev. E 98, 062139 (2018).
        """
        try:
            log_msd = np.log(self.empirical_msd)
            log_n = np.array([np.log(i) for i in range(1,self.max_number_of_points_in_msd)])
            alpha = ((self.max_number_of_points_in_msd+1) * np.sum(log_n * log_msd) - np.sum(log_n * np.sum(log_msd))) / \
                    ((self.max_number_of_points_in_msd+1) * np.sum(log_n ** 2) - (np.sum(log_n)) ** 2)
        except ValueError:
            alpha = 0
        return alpha

    
    def estimate_diffusion_exponent_with_noise_2(self):
        """
        The estimation of diffusion exponent with noise, according to method II from
        Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska,
        "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data"
        Phys. Rev. E 98, 062139 (2018).
        """    
        s2_max = self.empirical_msd[0]
        alpha_0 = 1
        D_0 = self.empirical_msd[0]
        s2_0 = self.empirical_msd[0]/2
        eps = 0.001
        
        try:
            popt, cov = curve_fit(
                    lambda x, D, a, s2: generate_theoretical_msd_anomalous_with_noise(x, D, self.dt, a, s2, self.dim),
                    self.n_list, self.empirical_msd, p0 =(D_0,alpha_0,s2_0) , bounds=([0,0,0],[np.inf,2,s2_max]), 
                    method = 'trf', ftol = eps)                   
            D_est = popt[0]
            alpha_est = popt[1]
        except ValueError:
            D_est = 0
            alpha_est = 0
        
        return alpha_est, D_est
    
    def estimate_diffusion_exponent_with_noise_3(self):
        """
        The estimation of diffusion exponent with noise, according to method III from with n_min fixed from
        Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska,
        "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data"
        Phys. Rev. E 98, 062139 (2018).
        """          
        alpha_0 = 1
        D_0 = self.empirical_msd[0]
        eps = 0.01

        def mds_fitting(n_list,de,dt,al):
            r = 4 * de * dt ** al * (n_list - 1) ** al
            return r

        popt, cov = curve_fit(
                        lambda x, D, a: mds_fitting(x, D, self.dt, a),
                        self.n_list, self.empirical_msd, p0 =(D_0,alpha_0), bounds=([0,0],[np.inf,2]), 
                        method = 'dogbox', ftol = eps)   
        
        D_est = popt[0]
        alpha_est = popt[1]
        
        return alpha_est, D_est
    
    
class CharacteristicFive(CharacteristicFour):
    """
    Experimental class for 
    """

    def __init__(self, x, y, z, dim, file, percentage_max_n=0.1, typ="", motion=""):
        """
        :param x: list, x coordinates
        :param y: list, y coordinates
        :param y: list, z coordinates
        :param dim: int, dimension
        :param file: str, path to trajectory
        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
        :param typ: str, type of diffusion i.e sub, super, rand
        :param motion: str, mode of diffusion eg. normal, directed
        """

        CharacteristicFour.__init__(self, x, y, z, dim, file, percentage_max_n, typ, motion)
        
        self.velocity_autocorrelation, self.velocity_autocorrelation_names = self.get_velocity_autocorrelation([1,2])
        self.ksstat = self.get_ksstat()        
        self.dagostino_stats, self.dagostino_stats_names = self.get_dagostino_stats()
        self.mv_features, self.mv_features_names = self.moving_window(windows=[10,20])
        self.eM, self.eL, self.eJ = self.get_exponents()
        self.maximum_ts = self.get_maximum_test_statistic()
        self.radius_gyration_tensor = self.get_tensor()
        self.trappedness = self.get_trappedness()
        self.eigenvalues, self.eigenvectors = LA.eig(self.radius_gyration_tensor)
        self.asymmetry = self.get_asymmetry()
        self.diff_kurtosis = self.get_kurtosis_corrected()
        self.efficiency = self.get_efficiency()
        
        self.max_std_x, self.max_std_y = self.max_min_std()
        self.max_std_change_x, self.max_std_change_y = self.max_std_change()
        self.velocity_autocorrelation, self.velocity_autocorrelation_names = self.get_velocity_autocorrelation([1,2])
        self.dma, self.dma_names = self.get_dma([1,2])
        
        self.values = [self.file, self.type, self.motion, self.D, self.alpha,
                       self.alpha_n_1, self.alpha_n_2, self.alpha_n_3, 
                       self.fractal_dimension, self.mean_gaussianity,
                       self.mean_squared_displacement_ratio, self.straightness,
                       self.p_variation, self.max_excursion_normalised, self.ksstat,
                       self.eM, self.eL, self.eJ, self.maximum_ts, self.trappedness, 
                       self.asymmetry, self.diff_kurtosis, self.efficiency,
                       self.max_std_x, self.max_std_y, self.max_std_change_x, self.max_std_change_y] \
                       + list(self.velocity_autocorrelation) \
                       + list(self.p_variations) + self.dagostino_stats + self.mv_features + list(self.dma)
                       

        self.columns = ["file", "Alpha", "motion", "D", "alpha",
                        "alpha_n_1", "alpha_n_2", "alpha_n_3", 
                        "fractal_dimension", "mean_gaussianity",
                        "mean_squared_displacement_ratio", "straightness",
                        "p-variation", "max_excursion_normalised", "ksstat_chi2",
                        "M", "L", "J", "max_ts", "trappedness", 'asymmetry', 
                        "diff_kurtosis", "efficiency", 'max_std_x', 'max_std_y', 
                        'max_std_change_x', 'max_std_change_y'] + self.velocity_autocorrelation_names \
                        + self.p_variation_names + self.dagostino_stats_names + self.mv_features_names \
                        + list(self.dma_names)
                        
        self.data = pd.DataFrame([self.values], columns=self.columns)
        
    def get_ksstat(self):
        
        dx = np.diff(self.x)
        dxn = (dx - np.nanmean(dx))/np.nanstd(dx)
        distpl = dxn**2
        if self.dim > 1:
            dy = np.diff(self.y)
            dyn = (dy - np.nanmean(dy))/np.nanstd(dy)
            distpl = distpl + dyn**2
        if self.dim >2 :
            dz = np.diff(self.z)
            dzn = (dz - np.nanmean(dz))/np.nanstd(dz)
            distpl = distpl + dzn**2
            
        ts = np.linspace(min(distpl),max(distpl),len(distpl))
        ts_nonzero = np.where(ts==0, np.finfo(float).eps, ts)
                    
        [stat,pv] = kstest(distpl, 'chi2', args=(ts_nonzero,self.dim), alternative='two-sided', mode='exact')
    
        return stat

    def get_dagostino_stats(self):
        
        stats, names = [], []
        
        stat, p = normaltest(np.diff(self.x))
        stats.append(stat)
        names.append('dagostino_x')
        if self.dim > 1:
            stat, p = normaltest(np.diff(self.y))
            stats.append(stat)
            names.append('dagostino_y')
        if self.dim >2 :
            stat, p = normaltest(np.diff(self.z))
            stats.append(stat)
            names.append('dagostino_z')
        
        return stats, names
    
    def moving_window(self, windows):
        
        values = []
        full_names =[]
        for window in windows:
            names = ['mw_x_mean','mw_y_mean','mw_x_std','mw_y_std']
            names = [st + str(window) for st in names]
            
            while self.N < window+2:
                window = int(self.N/2)
            
            mvw_x_mean = pd.Series(self.x).rolling(window=window).mean()
            xmd = np.nanmean(np.abs(np.diff(np.sign(np.diff(mvw_x_mean)))))/2
            mvw_x_std = pd.Series(self.x).rolling(window=window).std()
            xsd = np.nanmean(np.abs(np.diff(np.sign(np.diff(mvw_x_std)))))/2
            
            mvw_y_mean = pd.Series(self.y).rolling(window=window).mean()
            ymd = np.nanmean(np.abs(np.diff(np.sign(np.diff(mvw_y_mean)))))/2
            mvw_y_std = pd.Series(self.y).rolling(window=window).std()
            ysd = np.nanmean(np.abs(np.diff(np.sign(np.diff(mvw_y_std)))))/2
            
            values = values + [xmd, ymd, xsd, ysd]
            full_names = full_names + names
        
        return values, full_names
    
    def get_exponents(self):
        
        if self.dim==1:
            onelong = self.x
        elif self.dim==2:
            onelong = np.concatenate([self.x, self.y])
        if self.dim==3:
            onelong = np.concatenate([self.x, self.y, self.z])
        
        return FindExponents(onelong, self.dim)
    
    def get_trappedness(self, n=2):
        """
        Trappedness is the probability that a diffusing particle with the diffusion coefficient D
        and traced for a time interval t is trapped in a bounded region with radius r0.
        :param n: int, given point of trappedness
        :return: float, probability of trappedness in point n
        """
        t = self.n_list * self.dt
        popt, _ = curve_fit(lambda x, d: generate_theoretical_msd_normal(self.n_list[:2], d, self.dt, self.dim),
                            self.n_list[:2], self.empirical_msd[:2])
        d = popt[0]
        p = 1 - np.exp(0.2048 - 0.25117 * ((d * t) / (self.d / 2) ** 2))
        p = np.array([i if i > 0 else 0 for i in p])[n]
        
        return p
    
    def get_maximum_test_statistic(self):
        """
        :return: float, the value of the maximum test statistics
        """
        distance = np.array(
            [self.get_displacement(self.x[i], self.y[i], self.x[0], self.y[0]) for i in range(1, self.N)])
        d_max = np.max(distance)
        # TODO: The sigma estimator can be improved (Briane et al., 2018)
        sigma_2 = 1 / (2 * (self.N - 1) * self.dt) * np.sum(self.displacements ** 2)
        ts = d_max / np.sqrt(sigma_2 * self.T)

        return ts
    
    def get_tensor(self):
        """
        :return: matrix, the tensor T for given trajectory
        """
        a = sum((self.x - np.mean(self.x)) ** 2) / len(self.x)
        c = sum((self.y - np.mean(self.y)) ** 2) / len(self.y)
        b = sum((self.x - np.mean(self.x)) * (self.y - np.mean(self.y))) / len(self.x)
        return np.array([[a, b], [b, c]])
        
    def get_asymmetry(self):
        """
        The asymmetry of a trajectory can be used to detect directed motion.
        :return: float, asymmetry parameter - only real part of
        """
        lambda1 = self.eigenvalues[0]
        lambda2 = self.eigenvalues[1]
        a = -1 * np.log(1 - (lambda1 - lambda2) ** 2 / (2 * (lambda1 + lambda2) ** 2))
        return a.real
    
    def get_kurtosis_corrected(self):
        """
        Kurtosis measures the asymmetry and peakedness of the distribution of points within a trajectory
        :return: float, kurtosis for trajectory
        """
        index = np.where(self.eigenvalues == max(self.eigenvalues))[0][0]
        dominant_eigenvector = self.eigenvectors[index]
        a_prod_b = np.array([sum(np.array([self.x[i], self.y[i]]) * dominant_eigenvector) for i in range(len(self.x))])
        K = 1 / self.N * sum((a_prod_b - np.mean(a_prod_b)) ** 4 / np.std(a_prod_b) ** 4) - 3
        return K
    
    def get_efficiency(self):
        """
        Efficiency relates the net squared displacement of a particle to the sum of squared step lengths
        :return: float, efficiency parameter
        """
        upper = self.get_displacement(self.x[self.N - 2], self.y[self.N - 2], self.x[0], self.y[0]) ** 2
        displacements_to_squere = self.displacements ** 2
        lower = (self.N - 1) * sum(displacements_to_squere)
        E = upper / lower
        return E
    
#class CharacteristicExtension(CharacteristicBase):
#    
#    def __init__(self, x, y, z, dim, file, percentage_max_n=0.1, typ="", motion=""):
#        """
#        :param x: list, x coordinates
#        :param y: list, y coordinates
#        :param y: list, z coordinates
#        :param dim: int, dimension
#        :param file: str, path to trajectory
#        :param percentage_max_n: float, percentage of length of the trajectory for msd generating
#        :param typ: str, type of diffusion i.e sub, super, rand
#        :param motion: str, mode of diffusion eg. normal, directed
#        """
#
#        CharacteristicBase.__init__(self, x, y, z, dim, file, percentage_max_n, typ, motion)
#        

        
    def max_min_std(self, window=3):
        stds_x = pd.Series(self.x).rolling(window=window).std()
        stds_y = pd.Series(self.y).rolling(window=window).std()
    
        ratio_x = np.nanmax(stds_x)/np.nanmean(stds_x)
        ratio_y = np.nanmax(stds_y)/np.nanmean(stds_y)
    
        return ratio_x, ratio_y
    
    def max_std_change(self, window=3):
        stds_x = pd.Series(self.x).rolling(window=window).std()
        stds_y = pd.Series(self.y).rolling(window=window).std()
    
        change_x = np.nanmax(np.abs(np.diff(stds_x)))/np.std(self.x)
        change_y = np.nanmax(np.abs(np.diff(stds_y)))/np.std(self.y)
    
        return change_x, change_y
            
    def get_velocity_autocorrelation(self, hc_lag_list):
        """
        Calculate the velocity autocorrelation
        :return: float, the empirical autocorrelation for lag 1.
        """
        # hc_lag_list = [1,2,3,4,5]
        titles = ["vac_lag_" + str(x) for x in hc_lag_list]
        if self.dim == 1:
            autocorr = generate_empirical_velocity_autocorrelation(self.x, np.zeros(len(self.x)), hc_lag_list, self.dt, delta=1)
        elif self.dim == 2:
            autocorr = generate_empirical_velocity_autocorrelation(self.x, self.y, hc_lag_list, self.dt, delta=1)
        return autocorr, titles
    
    def get_dma(self, lag_list):
        
        titles = ["dma_lag_" + str(x) for x in lag_list]
        if self.dim == 1:
            dma = generate_detrended_moving_average(self.x, np.zeros(len(self.x)), lag_list)
        elif self.dim == 2:
            dma = generate_detrended_moving_average(self.x, self.y, lag_list)
        return dma, titles
