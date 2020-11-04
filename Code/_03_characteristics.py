import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from _02_msd import generate_theoretical_msd_normal, generate_empirical_msd, \
                    generate_theoretical_msd_anomalous_log, generate_empirical_pvariation, \
                    generate_empirical_velocity_autocorrelation, \
                    generate_theoretical_msd_anomalous_with_noise
                    
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
        popt, _ = curve_fit(lambda x, d: generate_theoretical_msd_normal(x, d, self.dt), self.n_list,
                            self.empirical_msd)
        D = popt[0]
        return D

    def get_exponent_alpha(self):
        """
        :return: float, exponential anomalous parameter by alpha parameter;
        estimated based on curve fitting of empirical and normal anomalous diffusion.
        Modification of this function can also estimate D parameter
        """
        popt, _ = curve_fit(
            lambda x, log_D, a: generate_theoretical_msd_anomalous_log(np.log(self.dt * self.n_list), log_D, a),
            np.log(self.dt * self.n_list), np.log(self.empirical_msd), bounds=((-np.inf, 0), (np.inf, 2)))
        alpha = popt[1]
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
        self.columns = ["file", "diff_type", "motion", "D", "alpha",
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
            gamma_power_fit = LinearRegression().fit(np.log(m_list).reshape(-1, 1), np.log(pv))
            gamma = gamma_power_fit.coef_[0]
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
        excursion = self.d / self.get_total_displacement()
        return excursion
        
        
    def estimate_diffusion_exponent_with_noise_1(self):
        """
        The estimation of diffusion exponent with noise, according to method I from
        Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska,
        "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data"
        Phys. Rev. E 98, 062139 (2018).
        """
        log_msd = np.log(self.empirical_msd)
        log_n = np.array([np.log(i) for i in range(1,self.max_number_of_points_in_msd)])
        alpha = ((self.max_number_of_points_in_msd+1) * np.sum(log_n * log_msd) - np.sum(log_n * np.sum(log_msd))) / \
                ((self.max_number_of_points_in_msd+1) * np.sum(log_n ** 2) - (np.sum(log_n)) ** 2)
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
        
        popt, cov = curve_fit(
                lambda x, D, a, s2: generate_theoretical_msd_anomalous_with_noise(x, D, self.dt, a, s2),
                self.n_list, self.empirical_msd, p0 =(D_0,alpha_0,s2_0) , bounds=([0,0,0],[np.inf,2,s2_max]), 
                method = 'trf', ftol = eps)   
                
        D_est = popt[0]
        alpha_est = popt[1]
        
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
    