import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import obspy
import sys
sys.path.append('./')
from phasecorr.phasecorr import acorr
from phasecorr.phasecorr import xcorr
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.stats import entropy
import matplotlib

def phasor_walkout(xi, dt, f_target):
    """
    Function to calculate phasor walkout for a given time series
    xi: input time series
    dt: sampling interval
    f_target: target frequency for phasor walkout
    """
    N = len(xi)
    ws = 2 * np.pi * f_target
    phasors = xi * np.exp(-1j * ws * np.arange(N) * dt)

    return phasors, np.sum(phasors)

def summation_dial(xi, dt, f_target, m):
    """
    Function to calculate summation dial of the phasor walkout for a given time series
    xi: input time series
    dt: sampling interval
    f_target: target frequency for phasor walkout
    m: dial size
    """
    N = len(xi)
    phasors, _ = phasor_walkout(xi, dt, f_target)
    intermediate_sums = [np.sum(phasors[i:i+m]) for i in range(0, N, m)]
    return intermediate_sums, np.sum(intermediate_sums)

def visualize_walkout(xi, dt, f_target, save_fig=False, fig_index=0):
    """
    Function to visualize phasor walkout for a given time series
    xi: input time series
    dt: sampling interval
    f_target: target frequency for phasor walkout
    """
    phasors, _ = phasor_walkout(xi, dt, f_target)
    cumulative_sums = np.cumsum(phasors)
    
    # Arrow position calculations
    x_pos = np.insert(cumulative_sums.real[:-1], 0, 0)
    y_pos = np.insert(cumulative_sums.imag[:-1], 0, 0)
    
    plt.figure(figsize=(10, 10))
    plt.axis('equal')
    plt.quiver(x_pos, y_pos, np.diff(cumulative_sums.real, prepend=0), np.diff(cumulative_sums.imag, prepend=0), angles='xy', scale_units='xy', scale=1, color='b')
    plt.xlabel('Real', fontsize=16)
    plt.ylabel('Imaginary', fontsize=16)
    plt.title('Phasor Walkout for f = {:} Hz ({:.2f} s in period)'.format(f_target, 1/f_target))
    plt.grid(True)
    if save_fig:
        plt.savefig('phasor_walkout_' + str(fig_index) + '.png')
    plt.show()

def visualize_walkout_for_subplot(xi, dt, f_target):
    """
    Function to visualize phasor walkout for a given time series
    xi: input time series
    dt: sampling interval
    f_target: target frequency for phasor walkout
    """
    phasors, _ = phasor_walkout(xi, dt, f_target)
    cumulative_sums = np.cumsum(phasors)
    
    # Arrow position calculations
    x_pos = np.insert(cumulative_sums.real[:-1], 0, 0)
    y_pos = np.insert(cumulative_sums.imag[:-1], 0, 0)
    
    # use a colormap to color the arrows based on their position on the time series
    colors = np.arange(0, len(xi)*dt, dt)

    plt.quiver(x_pos, y_pos, np.diff(cumulative_sums.real, prepend=0), np.diff(cumulative_sums.imag, prepend=0), colors, cmap='viridis', angles='xy', scale_units='xy', scale=1)
    # extend the figure to leave more space around the arrows
    plt.xlim(-2.6*np.max(np.abs(cumulative_sums.real)), 1.3*np.max(np.abs(cumulative_sums.real)))
    plt.ylim(-2.6*np.max(np.abs(cumulative_sums.imag)), 1.3*np.max(np.abs(cumulative_sums.imag)))
    
    plt.axis('equal')
    #plt.quiver(x_pos, y_pos, np.diff(cumulative_sums.real, prepend=0), np.diff(cumulative_sums.imag, prepend=0), angles='xy', scale_units='xy', scale=1, color='b')
    plt.xlabel('Real', fontsize=16)
    plt.ylabel('Imaginary', fontsize=16)
    plt.gca().ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize(14)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.grid(True)
    
    return

def visualize_walkout_for_subplot_sum_with_period(xi, dt, f_target, noramlize_each_period=True):
    """
    Function to visualize phasor walkout for a given time series
    xi: input time series
    dt: sampling interval
    f_target: target frequency for phasor walkout
    """
    phasors, _ = phasor_walkout(xi, dt, f_target)
    
    N = len(xi)
    # sum every period
    samples_per_period = 1 / (f_target * dt)
    period_indices = np.arange(0, N, samples_per_period).astype(int)
    period_indices = period_indices[period_indices + samples_per_period + 1 <= N]
    phasors = [np.sum(phasors[i:i+int(samples_per_period)]) for i in period_indices]
    
    if noramlize_each_period:
        phasors = [phasor/np.abs(phasor) for phasor in phasors]

    cumulative_sums = np.cumsum(phasors)
    
    # Arrow position calculations
    x_pos = np.insert(cumulative_sums.real[:-1], 0, 0)
    y_pos = np.insert(cumulative_sums.imag[:-1], 0, 0)
    
    # use a colormap to color the arrows based on their position on the time series
    colors = period_indices

    plt.quiver(x_pos, y_pos, np.diff(cumulative_sums.real, prepend=0), np.diff(cumulative_sums.imag, prepend=0), colors, cmap='viridis', angles='xy', scale_units='xy', scale=1)
    # extend the figure to leave more space around the arrows
    plt.xlim(-2.6*np.max(np.abs(cumulative_sums.real)), 1.3*np.max(np.abs(cumulative_sums.real)))
    plt.ylim(-2.6*np.max(np.abs(cumulative_sums.imag)), 1.3*np.max(np.abs(cumulative_sums.imag)))
    
    plt.axis('equal')
    #plt.quiver(x_pos, y_pos, np.diff(cumulative_sums.real, prepend=0), np.diff(cumulative_sums.imag, prepend=0), angles='xy', scale_units='xy', scale=1, color='b')
    plt.xlabel('Real', fontsize=16)
    plt.ylabel('Imaginary', fontsize=16)
    plt.gca().ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize(14)
    plt.gca().yaxis.get_offset_text().set_fontsize(14)
    plt.grid(True)
    
    return

def linearity_measure_by_sum_ratio(xi, dt, f_target):
    phasors, _ = phasor_walkout(xi, dt, f_target)
    N = len(xi)

    # sum every period
    samples_per_period = 1 / (f_target * dt)
    period_indices = np.arange(0, N, samples_per_period).astype(int)
    period_indices = period_indices[period_indices + samples_per_period + 1 <= N]
    phasors = [np.sum(phasors[i:i+int(samples_per_period)]) for i in period_indices]

    resultant_vector = np.sum(phasors)
    
    # Compute the magnitude of the resultant vector
    resultant_magnitude = np.abs(resultant_vector)
    
    # Compute the sum of the magnitudes of individual vectors
    sum_of_magnitudes = np.sum([np.abs(c) for c in phasors])
    
    # Compute the linearity ratio
    ratio = resultant_magnitude / sum_of_magnitudes
    
    return ratio

def linearity_measure_by_sum_ratio_unit(xi, dt, f_target):
    phasors, _ = phasor_walkout(xi, dt, f_target)
    N = len(xi)
    samples_per_period = 1 / (f_target * dt)
    period_indices = np.arange(0, N, samples_per_period).astype(int)
    period_indices = period_indices[period_indices + samples_per_period + 1 <= N]
    phasors = [np.sum(phasors[i:i+int(samples_per_period)]) for i in period_indices]

    resultant_vector = np.sum(phasors)
    
    # Compute the magnitude of the resultant vector
    resultant_magnitude = np.abs(resultant_vector)
    
    # Compute the sum of the magnitudes of individual vectors
    sum_of_magnitudes = np.sum([np.abs(c) for c in phasors])
    
    # Compute the linearity ratio
    ratio = resultant_magnitude / sum_of_magnitudes
    
    return ratio

def visualize_dial(xi, dt, f_target, m):
    intermediate_sums, _ = summation_dial(xi, dt, f_target, m)
    cumulative_sums = np.cumsum(intermediate_sums)
    
    # Arrow position calculations
    x_pos = np.insert(cumulative_sums.real[:-1], 0, 0)
    y_pos = np.insert(cumulative_sums.imag[:-1], 0, 0)
    
    plt.figure(figsize=(10, 10))
    plt.axis('equal')
    plt.quiver(x_pos, y_pos, np.diff(cumulative_sums.real, prepend=0), np.diff(cumulative_sums.imag, prepend=0), angles='xy', scale_units='xy', scale=1, color='b')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Summation Dial for f = ' + str(f_target) + ' and m = ' + str(m))
    plt.grid(True)
    plt.show()

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs # nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # data: input signal
    # lowcut: low cut-off frequency
    # highcut: high cut-off frequency
    # fs: sampling frequency
    # order: order of the filter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def calculate_walkout_slope_R(xi, dt, f_target):
    """
    Function to calculate R value for a given complex array
    """
    phasors, _ = phasor_walkout(xi, dt, f_target)
    N = len(xi)
    samples_per_period = 1 / (f_target * dt)
    period_indices = np.arange(0, N, samples_per_period).astype(int)
    period_indices = period_indices[period_indices + samples_per_period + 1 <= N]
    phasors = [np.sum(phasors[i:i+int(samples_per_period)]) for i in period_indices]
    cumulative_sums = np.cumsum(phasors)
    
    x_pos = np.insert(cumulative_sums.real[:-1], 0, 0)
    y_pos = np.insert(cumulative_sums.imag[:-1], 0, 0)
    slope, _, r_value, _, _ = linregress(x_pos, y_pos)
    return slope, r_value

def calculate_walkout_R2(xi, dt, f_target):
    """
    Function to calculate R^2 value for a given complex array
    """
    phasors, _ = phasor_walkout(xi, dt, f_target)
    N = len(xi)
    samples_per_period = 1 / (f_target * dt)
    period_indices = np.arange(0, N, samples_per_period).astype(int)
    period_indices = period_indices[period_indices + samples_per_period + 1 <= N]
    phasors = [np.sum(phasors[i:i+int(samples_per_period)]) for i in period_indices]
    cumulative_sums = np.cumsum(phasors)
    
    x_pos = np.insert(cumulative_sums.real[:-1], 0, 0)
    y_pos = np.insert(cumulative_sums.imag[:-1], 0, 0)
    # linear model
    slope, intercept, r_value, _, _ = linregress(x_pos, y_pos)
    # predict y values
    y_pred = slope*x_pos + intercept
    # calculate R^2 value 1 - (SS_res / SS_tot)
    SS_res = np.sum((y_pos - y_pred)**2)
    SS_tot = np.sum((y_pos - np.mean(y_pos))**2)
    R2 = 1 - (SS_res / SS_tot)
    
    return R2

def schuster_test_for_phasor_walkout(xi, dt, f_target):
    """
    Function to calculate Schuster test for a given time series following Heaton, 1982 and Lognonne et al., 2023 
    xi: input time series
    dt: sampling interval
    f_target: target frequency for phasor walkout
    """
    phasors, _ = phasor_walkout(xi, dt, f_target)
    # sum every period
    N = len(xi)
    samples_per_period = 1 / (f_target * dt)
    period_indices = np.arange(0, N, samples_per_period).astype(int)
    period_indices = period_indices[period_indices + samples_per_period + 1 <= N]
    phasors = [np.sum(phasors[i:i+int(samples_per_period)]) for i in period_indices]

    # normalize the phasors to unit length
    phasors = [phasor/np.abs(phasor) for phasor in phasors]

    resultant_vector = np.sum(phasors)

    schuster_p = np.exp( (-1.0)*(np.abs(resultant_vector)**2) / len(phasors) )
    schuster_significance = 1 - schuster_p
    epillion = 1e-50
    ss_log = -1.0*np.log10(schuster_p + epillion)
    return schuster_significance, ss_log

def calculate_KL_divergence(Qi, Pi):
    """
    Function to calculate KL divergence between two probability distributions
    Qi: first probability distribution
    Pi: second probability distribution
    """
    return entropy(Qi, Pi)