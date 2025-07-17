from phw_lib import *

def get_pdf_prob(X, N, std=1):
    """
    Probability density function for the normalized Gaussian distribution.
    """
    return (2*np.abs(X) / (N*(std**2)) )* np.exp(-X**2 / (N*(std**2)))

res_dict_noise_time_duration = {}

fs = 6.6
dt = 1/fs
test_num = 100

time_duration_list = np.arange(10*1e4, 42*1e4, 2*1e4)

#time_duration_list = [12000]

for time_duration in time_duration_list:
    n_samples = int(time_duration/dt)
    t = np.arange(0, time_duration, dt)
    freqs = np.fft.fftfreq(len(t), dt)

    print('On Time duration {:}'.format(time_duration))

    # bandpass filter
    lower_freq = 2.5*1e-3
    upper_freq = 12.5*1e-3

    search_lower_range = 3*1e-3
    search_upper_range = 12*1e-3

    # calculate R_value for all peaks for the noise + sin
    R_values_noise = []
    Sum_ratio_values_noise = []
    Schuster_values_noise = []
    SS_mag_values_noise = []
    pdf_prob_noise = []


    for test_dx in range(test_num):
        noise = np.random.normal(0, 1, size=n_samples)

        # find the top 100 peaks within the frequency range
        fft_amp = np.abs(np.fft.fft(noise))
        peaks, _ = find_peaks(fft_amp, distance=10)
        peaks = peaks[np.logical_and(freqs[peaks] > search_lower_range, freqs[peaks] < search_upper_range)]
        # remove the peak at the true frequency sin_wave_freq
        peaks = peaks[np.argsort(fft_amp[peaks])][::-1][:20]
        R_values_noise.append([calculate_walkout_slope_R(noise, dt, freqs[peak])[1] for peak in peaks])
        Sum_ratio_values_noise.append([linearity_measure_by_sum_ratio(noise, dt, freqs[peak]) for peak in peaks])
        for peak in peaks:
            SS, SS_mag = schuster_test_for_phasor_walkout(noise, dt, freqs[peak])
            Schuster_values_noise.append(SS)
            SS_mag_values_noise.append(SS_mag)
            pdf_prob_noise.append(get_pdf_prob(fft_amp[peak], n_samples))



    res_dict_noise_time_duration[time_duration] = {'R_values_noise': R_values_noise,
                                                    'Sum_ratio_values_noise': Sum_ratio_values_noise,
                                                    'Schuster_values_noise': Schuster_values_noise,
                                                    'SS_mag_values_noise': SS_mag_values_noise,
                                                    'pdf_prob_noise': pdf_prob_noise}

# save the results in npz file
np.savez('./Seq_Length_Study_SS_mag.npz', res_dict_noise_time_duration=res_dict_noise_time_duration)