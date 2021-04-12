import numpy as np

"""
The helper functions used in characteristics calculation.
"""

def generate_theoretical_msd_normal(n_list, D, dt, dim):
    """
    Function for generating msd of normal diffusion
    :param n_list: number of points in msd
    :param D: float, diffusion coefficient
    :param dt: float, time between steps
    :param dim: int, dimension (1,2,3)
    :return: array of theoretical msd
    """
    r = 2 * dim * D * dt * n_list
    return r

def generate_theoretical_msd_anomalous_log(log_dt_n_list, log_D, alpha, dim):
    """
    Function for generating logarithm msd of anomalous diffusion
    :param log_dt_n_list: logarithm of points in msd times dt
    :param log_D: float, logarithm of diffusion coefficient   
    :param alpha: float, anomalous exponent (alpha<1)
    :param dim: int, dimension (1,2,3)
    :return: array of logarithm theoretical msd
    """
    r = np.log(2 * dim) + log_D + alpha * log_dt_n_list
    return r

def generate_theoretical_msd_anomalous_with_noise(n_list, D, dt, alpha, sigma_2, dim):
    """
    Function for generating msd of anomalous diffusion
    :param n_list: number of points in msd
    :param D: float, diffusion coefficient
    :param dt: float, time between steps
    :param alpha: float, anomalous exponent (alpha<1)
    :param sigma_2: float, noise
    :param dim: int, dimension (1,2,3)
    :return: array of theoretical msd
    """
    r = 2 * dim * D * (dt * n_list) ** alpha + sigma_2
    return r

def generate_empirical_msd(x, y, n_list, k=2):
    """
    Function for generating empirical msd for a list of lags
    :param x: list, list of x coordinates
    :param y: list, list of y coordinates
    :param n_list: number of points in msd
    :param k: int, power of msd
    :return: array of empirical msd
    """
    r = []
    if type(y) == type(None):
        y = np.zeros(len(x))
    for n in n_list:
        r.append(empirical_msd(x, y, n, k))
    return np.array(r)


def empirical_msd(x, y, n, k):
    """
    Function for generating empirical msd for a single lag
    :param x: list, list of x coordinates
    :param y: list, list of y coordinates
    :param n: int, point of msd
    :param k: int, power of msd
    :return: point of empirical msd for given point
    """
    N = len(x)
    x1 = np.array(x[:N - n])
    x2 = np.array(x[n:N])
    y1 = np.array(y[:N - n])
    y2 = np.array(y[n:N])
    c = np.sqrt(np.array(list(x2 - x1)) ** 2 + np.array(list(y2 - y1))**2) ** k
    r = np.mean(c)
    return r


def generate_empirical_pvariation(x, y, p_list=[2], m_list=[1]):
    """
    :param x: list, list of x coordinates
    :param y: list, list of y coordinates
    :param p_list: powers of p-variation, default p=[2] - quadratic variation
    :param m_list: the choice of lags, default m=[1] - simple differences
    :return: array of empirical pvariation
    """
    N = len(x)

    pvar = np.zeros((len(p_list), len(m_list)))
    for p_index in range(len(p_list)):
        for m_index in range(len(m_list)):
            p = p_list[p_index]
            m = m_list[m_index]
            sample_indexes = np.arange(0, N-m, m)
            x_diff = np.take(x, sample_indexes+m) - np.take(x, sample_indexes)
            y_diff = np.take(y, sample_indexes+m) - np.take(y, sample_indexes)
            pvar[p_index][m_index] = sum(np.sqrt(x_diff ** 2 + y_diff ** 2) ** p)
    return pvar

def generate_empirical_velocity_autocorrelation(x, y, n_list, dt, delta=1):
    """
    Function for generating empirical autocorrelation for the given list of points
    :param x: list, list of x coordinates
    :param y: list, list of y coordinates
    :param n_list: list, number of points in autocorrelation
    :param dt: float, time between steps
    :param delta: the time lag between observations (default=1)
    :return: array of empirical autocorrelation
    """
    r = []
    for n in n_list:
        r.append(empirical_velocity_autocorrelation(x, y, n, dt, delta))
    return np.array(r)


def empirical_velocity_autocorrelation(x, y, n, dt, delta):
    """
    Function for generating empirical autocorrelation for single point
    :param x: list, list of x coordinates
    :param y: list, list of y coordinates
    :param n: int, point of autocorrelation
    :param dt: float, time between steps
    :param delta: the time lag between observations (default=1)
    :return: point of empirical msd for given point
    """

    velocities_x = np.diff(x, delta)/(delta*dt)
    velocities_y = np.diff(y, delta)/(delta*dt)
    N = len(velocities_x)

    vx1 = np.array(velocities_x[:N - n])
    vx2 = np.array(velocities_x[n:])
    vy1 = np.array(velocities_y[:N - n])
    vy2 = np.array(velocities_y[n:])

    c = vx2 * vx1 + vy2 * vy1
    r = np.mean(c)

    return r
