# This python script is based on the matlab code from http://www.neitzvision.com/img/research/spectsens.m

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.io import savemat, loadmat

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from root_config import ROOT_DIR

def generate_sensitivity_curve(lambda_max=559, OD=0.30, output='alog', spectrum=None):
    if spectrum is None:
        spectrum = np.arange(400, 701)

    A, B, C, D = 0.417050601, 0.002072146, 0.000163888, -1.922880605
    E, F, G, H = -16.05774461, 0.001575426, 5.11376E-05, 0.00157981
    I, J, K, L = 6.58428E-05, 6.68402E-05, 0.002310442, 7.31313E-05
    M, N, O, P = 1.86269E-05, 0.002008124, 5.40717E-05, 5.14736E-06
    Q, R, S, T = 0.001455413, 4.217640000E-05, 4.800000000E-06, 0.001809022
    U, V, W, X = 3.86677000E-05, 2.99000000E-05, 0.001757315, 1.47344000E-05
    Y, Z = 1.51000000E-05, OD + 1E-8

    A2 = np.log10(1.0 / lambda_max) - np.log10(1.0 / 558.5)
    vector = np.log10(1.0 / spectrum)

    const = 1 / np.sqrt(2 * np.pi)
    ex_temp1 = np.log10(-E + E * np.tanh(-((10 ** (vector - A2)) - F) / G)) + D
    ex_temp2 = A * np.tanh(-((10 ** (vector - A2)) - B) / C)
    ex_temp3 = -(J / I * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - H) / I) ** 2)))
    ex_temp4 = -(M / L * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - K) / L) ** 2)))
    ex_temp5 = -(P / O * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - N) / O) ** 2)))
    ex_temp6 = (S / R * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - Q) / R) ** 2)))
    ex_temp7 = ((V / U * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - T) / U) ** 2))) / 10)
    ex_temp8 = ((Y / X * (const * np.exp(-0.5 * (((10 ** (vector - A2)) - W) / X) ** 2))) / 100)

    ex_temp = (ex_temp1 + ex_temp2 + ex_temp3 + ex_temp4 +
               ex_temp5 + ex_temp6 + ex_temp7 + ex_temp8)

    OD_temp = np.log10((1 - 10 ** -((10 ** ex_temp) * Z)) / (1 - 10 ** -Z))

    if output == 'log':
        return OD_temp
    else:
        return 10 ** OD_temp


def generate_lens_transmit(lambdas):
    # load the matlab file
    lens_data = loadmat(f"{ROOT_DIR}/Simulated/Retina/SS_spectral_sampling/helper/data/den_lens_ssf.mat")
    S_lens_ssf = np.arange(390, 831)
    lens_density = CubicSpline(S_lens_ssf, lens_data["den_lens_ssf"])(lambdas)
    lens_transmit = 10 ** (-lens_density)
    lens_transmit = lens_transmit[:,0]

    return lens_transmit


def generate_cone_fundamentals_from_peak_frequencies(peak_frequencies, lambdas=None):

    # lambda is the spectrum range (400-700 nm visible wavelength is the default)
    if lambdas is None:
        lambdas = np.arange(400, 701)

    # lens transmittance function (fixed throughout the cone fundamentals computation)
    lens_transmit = generate_lens_transmit(lambdas)

    cone_fundamentals = np.zeros((301, 4))
    for pid, peak_frequency in enumerate(peak_frequencies):
        cone_fundamentals[:, pid] = generate_sensitivity_curve(lambda_max=peak_frequency, OD=0.001, spectrum=lambdas)
        cone_fundamentals[:, pid] *= lens_transmit
        cone_fundamentals[:, pid] /= max(cone_fundamentals[:, pid])

    return cone_fundamentals