import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from scipy.stats import norm, kurtosis, skew
from scipy.signal import find_peaks, hilbert
from scipy.fft import fft, fftfreq
import time
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import torch
from sklearn.decomposition import PCA

def one_hot(x, class_name, num_class=4):
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), class_name-1] = 1
    return ohx

def demean_detrended_normalize_time_series(series):
    # Demean
    mean_val = np.mean(series)
    demeaned_series = series - mean_val
    # Detrend the demeaned series using a linear trend
    detrended_series = signal.detrend(demeaned_series)
    # norm to -1 and 1
    min_val = np.min(detrended_series)
    max_val = np.max(detrended_series)
    normalized_series = (detrended_series - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    normalized_series = normalized_series * 2 - 1  # Rescale to [-1, 1]
    return normalized_series

def load_time_seties_data(path):
    
    data_path = path
    class1_data_fname = 'whole_wave_class1.mat'
    class1_data_fname = data_path+class1_data_fname
    class1_data = sio.loadmat(class1_data_fname)
    class1_data = class1_data['whole_wave_class1']
    x_class1 = class1_data 
    y_class1 = one_hot(np.ones(x_class1.shape[0]), class_name=1)

    class2_data_fname = 'whole_wave_class2.mat'
    class2_data_fname = data_path+class2_data_fname
    class2_data = sio.loadmat(class2_data_fname)
    class2_data = class2_data['whole_wave_class2']
    x_class2 = class2_data 
    y_class2 = one_hot(np.ones(x_class2.shape[0]), class_name=2)

    class3_data_fname = 'whole_wave_class3.mat'
    class3_data_fname = data_path+class3_data_fname
    class3_data = sio.loadmat(class3_data_fname)
    class3_data = class3_data['whole_wave_class3']
    x_class3 = class3_data  
    y_class3 = one_hot(np.ones(x_class3.shape[0]), class_name=3)

    class4_data_fname = 'whole_wave_class4.mat'
    class4_data_fname = data_path+class4_data_fname
    class4_data = sio.loadmat(class4_data_fname)
    class4_data = class4_data['whole_wave_class4']
    x_class4 = class4_data  
    y_class4 = one_hot(np.ones(x_class4.shape[0]), class_name=4)

    x = []
    x = np.concatenate((x_class1, x_class2, x_class3, x_class4), axis=0) 
    y = []
    y = np.concatenate((y_class1, y_class2, y_class3, y_class4), axis=0) 

    return x, y

def load_time_seties_data_train_val(path, train_ratio=0.8):
    
    data_path = path
    class1_data_fname = 'whole_wave_class1.mat'
    class1_data_fname = data_path+class1_data_fname
    class1_data = sio.loadmat(class1_data_fname)
    class1_data = class1_data['whole_wave_class1']
    split_index_class1 = int(len(class1_data) * train_ratio)
    # Shuffle the data (optional, but recommended)
    np.random.shuffle(class1_data)
    # Split the data into training and testing sets
    train_class1_data, val_class1_data = class1_data[:split_index_class1], class1_data[split_index_class1:]
    train_class1_data_y_class1 = one_hot(np.ones(train_class1_data.shape[0]), class_name=1)
    val_class1_data_y_class1 = one_hot(np.ones(val_class1_data.shape[0]), class_name=1)

    class2_data_fname = 'whole_wave_class2.mat'
    class2_data_fname = data_path+class2_data_fname
    class2_data = sio.loadmat(class2_data_fname)
    class2_data = class2_data['whole_wave_class2']
    split_index_class2 = int(len(class2_data) * train_ratio)
    # Shuffle the data (optional, but recommended)
    np.random.shuffle(class2_data)
    # Split the data into training and testing sets
    train_class2_data, val_class2_data = class2_data[:split_index_class2], class2_data[split_index_class2:]
    train_class2_data_y_class2 = one_hot(np.ones(train_class2_data.shape[0]), class_name=2)
    val_class2_data_y_class2 = one_hot(np.ones(val_class2_data.shape[0]), class_name=2)

    class3_data_fname = 'whole_wave_class3.mat'
    class3_data_fname = data_path+class3_data_fname
    class3_data = sio.loadmat(class3_data_fname)
    class3_data = class3_data['whole_wave_class3']
    split_index_class3 = int(len(class3_data) * train_ratio)
    # Shuffle the data (optional, but recommended)
    np.random.shuffle(class3_data)
    # Split the data into training and testing sets
    train_class3_data, val_class3_data = class3_data[:split_index_class3], class3_data[split_index_class3:]
    train_class3_data_y_class3 = one_hot(np.ones(train_class3_data.shape[0]), class_name=3)
    val_class3_data_y_class3 = one_hot(np.ones(val_class3_data.shape[0]), class_name=3)

    class4_data_fname = 'whole_wave_class4.mat'
    class4_data_fname = data_path+class4_data_fname
    class4_data = sio.loadmat(class4_data_fname)
    class4_data = class4_data['whole_wave_class4']
    split_index_class4 = int(len(class4_data) * train_ratio)
    # Shuffle the data (optional, but recommended)
    np.random.shuffle(class4_data)
    # Split the data into training and testing sets
    train_class4_data, val_class4_data = class4_data[:split_index_class4], class4_data[split_index_class4:]
    train_class4_data_y_class4 = one_hot(np.ones(train_class4_data.shape[0]), class_name=4)
    val_class4_data_y_class4 = one_hot(np.ones(val_class4_data.shape[0]), class_name=4)

    x_train = []
    x_train = np.concatenate((train_class1_data, train_class2_data, train_class3_data, train_class4_data), axis=0)
    x_val = []
    x_val = np.concatenate((val_class1_data, val_class2_data, val_class3_data, val_class4_data), axis=0) 
    
    y_train = []
    y_train = np.concatenate((train_class1_data_y_class1, train_class2_data_y_class2, train_class3_data_y_class3, train_class4_data_y_class4), axis=0) 
    y_val = []
    y_val = np.concatenate((val_class1_data_y_class1, val_class2_data_y_class2, val_class3_data_y_class3, val_class4_data_y_class4), axis=0) 

    return x_train, x_val, y_train, y_val

def per_processing(input_array, axis): 

    # input_array = np.squeeze(input_array, axis=2)
    per_processing_array = np.zeros_like(input_array)
    max_val = np.max(input_array, axis, keepdims=True)
    min_val = np.min(input_array, axis, keepdims=True)

    # Step 2: Rescale to range [0, 1]
    input_array = (input_array - min_val) / (max_val - min_val+0.000000001)

    # Step 3: Rescale to range [-1, 1]
    per_processing_array = input_array * 2 - 1

    return per_processing_array


def bandpass_stft_filter(data, low_frequency=1, upper_frequency=20, fs=250):
    nyq_fre = 0.5*fs
    b, a = signal.butter(N = 4, Wn = [low_frequency/nyq_fre, upper_frequency/nyq_fre], btype='bandpass')  ## filter 1-20 Hz
    window = np.hamming(2500)
    filtered_array = np.zeros_like(data)
    x_stft_normalized_spectrogram = np.zeros(shape = [np.shape(data)[0], np.shape(data)[-1], 16, 48], dtype = None, order = 'C')
    norm_data = per_processing(data, axis=1)
    
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[-1]):
            x_win = norm_data[i,:,j] * window
            filtedData = signal.filtfilt(b, a, x_win)
            filtedData /= np.sum(np.hamming(2500))
            filtered_array[i,:,j] = filtedData
            ## stft 
            f, t, Zxx = signal.stft(filtedData, fs=250, nperseg = 128, noverlap = 118, nfft = 1024)  ## f max is 125
            normalized_spectrogram = preprocessing.normalize(np.abs(Zxx))
            aa = normalized_spectrogram[::32,::4]
            x_stft_normalized_spectrogram[i,j,:,:] = aa[0:16,2:50]
    filtered_array = per_processing(filtered_array, axis=1)
    return filtered_array, f, x_stft_normalized_spectrogram


def feature_construction(data, freq, x_stft):

    feature_array = np.zeros([np.shape(data)[0], np.shape(data)[-1], 55])
    for i_event in range(np.shape(data)[0]):
        print('event index', i_event)
        for j_channel in range(np.shape(data)[-1]):
            if np.isnan(data[i_event,:,j_channel]).any() or np.isnan(x_stft[i_event, j_channel,:,:]).any():
                j_channel = j_channel-1
            start_time = time.time()
            spectrogram = x_stft[i_event, j_channel,:,:]
            ## init feature value 
            ratio_mean_max_env, ratio_median_max_env, kurtosis_signal, kurtosis_env, Skewness_signal, Skewness_env, No_auto_peaks, energy_first_3rd, energy_remaining, ratio_f8_f9, max_env=[], [], [], [], [], [], [], [], [], [], []
            mean_DFT, max_DFT, max_frequency, centroid, first_quartile_frequency, second_quartile_frequency, median_normalized, Var_normalized, num_peaks_75_DFT, mean_peaks = [], [], [], [], [], [], [], [], [], []
            spectral_centroid, gyration_radius, centroid_width, Kurtosis_spec_function_t, Kurtosis_spec_function_f, Mean_ratio_max_mean, Mean_ratio_max_median, no_peaks_DFT_max, no_peaks_DFT_mean, no_peaks_DFT_median = [], [], [], [], [], [], [], [], [], []
            ratio_43_44, ratio_43_45, num_peak_spec_central_fre, num_peak_spec_max_f, ratio_48_49, mean_dis_max_mean_fre, mean_dis_max_median_fre, mean_distance_first_median, mean_distance_third_median, mean_distance_first_third = [], [], [], [], [], [], [], [], [], []
            ##
            analytic_signal = hilbert(data[i_event,:,j_channel])
            # Compute the magnitude of the analytic signal
            envh = np.abs(analytic_signal)
            mean_envh = np.mean(envh)
            median_envh = np.median(envh)
            max_envh = np.max(envh)
            ## feature 1 Ratio of the mean over the maximum of the envelope
            ratio_mean_max_env = mean_envh / max_envh
            ## feature 2 Ratio of the mean over the maximum of the envelope
            ratio_median_max_env = median_envh / max_envh
            ## feature 3 Kurtosis of the raw signal (peakness of the signal)
            kurtosis_signal = kurtosis(data[i_event,:,j_channel])
            ## feature 4 Kurtosis of the envelope
            kurtosis_env = kurtosis(envh)
            ## feature 5 Skewness of the raw signal
            Skewness_signal = skew(data[i_event,:,j_channel])
            ## feature 6 Skewness of the envelope
            Skewness_env = skew(envh)
            ## feature 7 Number of peaks in the autocorrelation function
            autocorr_values = np.correlate(data[i_event,:,j_channel], data[i_event,:,j_channel], mode='full')
            autocorr_values = autocorr_values[len(data[i_event,:,j_channel])-1:]
            autocorr_peaks, _ = find_peaks(autocorr_values)
            No_auto_peaks = len(autocorr_peaks)
            ## feature 8 Energy in the first third part of the autocorrelation function
            n = len(autocorr_values)
            first_third = n // 3
            energy_first_3rd = np.sum(autocorr_values[:first_third]**2)
            ## feature 9 Energy in the remaining part of the autocorrelation function
            energy_remaining = np.sum(autocorr_values[first_third+1:]**2)
            ## feature 10 Energy in the remaining part of the autocorrelation function
            ratio_f8_f9 = energy_first_3rd / energy_remaining
            ## feature 11-20 Energy of the signal filtered in 1 — 3 Hz, 3 — 6 Hz, 5 — 7 Hz, 6 — 9 Hz and 8 —10 Hz
            ## Kurtosis of the signal in 1 — 3 Hz, 3 — 6 Hz, 5 — 7 Hz, 6 — 9 Hz and 8—10 Hz frequency range
            fft_values = fft(data[i_event,:,j_channel])
            fft_values[0] = 0
            # Calculate the corresponding frequency values
            sampling_rate = 250  # Replace with your actual sampling rate
            freq_values = fftfreq(len(data[i_event,:,j_channel]), d=1/sampling_rate)
            bands = [(1, 6), (6, 12), (10, 14), (12, 18), (16, 20)]
            energy1_6hz = []
            energy6_12hz = []
            energy10_14hz = []
            energy12_18hz = []
            energy16_20hz = []
            Kurtosis1_6hz = []
            Kurtosis6_12hz = []
            Kurtosis10_14hz = []
            Kurtosis12_18hz = []
            Kurtosis16_20hz = []
            i_band=0
            for band in bands:
                start_freq, end_freq = band
                mask = (freq_values >= start_freq) & (freq_values <= end_freq)
                band_components = np.abs(fft_values[mask])
                band_energies = np.abs(fft_values[mask])**2
                if i_band==0:
                    energy1_6hz = np.sum(band_energies)
                    Kurtosis1_6hz = kurtosis(band_components)
                if i_band==1:
                    energy6_12hz = np.sum(band_energies)
                    Kurtosis6_12hz = kurtosis(band_components)
                if i_band==2:
                    energy10_14hz = np.sum(band_energies)
                    Kurtosis10_14hz = kurtosis(band_components)
                if i_band==3:
                    energy12_18hz = np.sum(band_energies)
                    Kurtosis12_18hz = kurtosis(band_components)
                if i_band==4:
                    energy16_20hz = np.sum(band_energies)
                    Kurtosis16_20hz = kurtosis(band_components)
                i_band=i_band+1
            ## feature 21 Maximum of the envelope
            max_env = max_envh

            #### sepctral features
            ## feature 22 Mean of the DFT
            mean_DFT = np.mean(np.abs(fft_values))
            ## feature 23 Max of the DFT
            max_DFT = np.max(np.abs(fft_values))
            ## feature 24 Frequency at the maximum
            max_frequency = freq_values[np.argmax(np.abs(fft_values))]
            ## feature 25 Frequency of spectrum centroid
            magnitude_spectrum = np.abs(fft_values)
            centroid = np.sum(freq_values * magnitude_spectrum) / np.sum(magnitude_spectrum)
            ## feature 26 Central frequency of the 1st quartile
            cumulative_distribution = np.cumsum(magnitude_spectrum) / np.sum(magnitude_spectrum)
            # Find the frequency at the first quartile
            quartile_index = np.argmax(cumulative_distribution >= 0.125)
            first_quartile_frequency = freq_values[quartile_index]
            ## feature 27 Central frequency of the 2rd quartile
            quartile_index = np.argmax(cumulative_distribution >= 0.25)
            second_quartile_frequency = freq_values[quartile_index]
            ## feature 28 Median of the normalized DFT
            # Normalize the DFT coefficients
            normalized_dft = fft_values / np.max(np.abs(fft_values))
            # Compute the median of the normalized DFT coefficients
            median_normalized = np.median(np.abs(normalized_dft))
            ## feature 29 Variance of the normalized DFT
            Var_normalized = np.var(np.abs(normalized_dft))
            ## feature 30 Number of peaks (> 0.75 DFTmax)
            # Find the maximum DFT coefficient
            magnitude_spectrum = np.abs(normalized_dft)
            max_dft = np.max(magnitude_spectrum)
            # Define the threshold
            threshold = 0.75 * max_dft
            # Count the number of peaks above the threshold
            num_peaks_75_DFT = 0
            for i_DFT in range(1, len(magnitude_spectrum) - 1):
                if magnitude_spectrum[i_DFT] > threshold and magnitude_spectrum[i_DFT] > magnitude_spectrum[i_DFT - 1] and magnitude_spectrum[i_DFT] > magnitude_spectrum[i_DFT + 1]:
                    num_peaks_75_DFT += 1
            ## feature 31 Number of peaks (> 0.75 DFTmax)
            peaks_value = [magnitude_spectrum[i] for i in range(1, len(magnitude_spectrum) - 1) if magnitude_spectrum[i] > threshold and magnitude_spectrum[i] > magnitude_spectrum[i - 1] and magnitude_spectrum[i] > magnitude_spectrum[i + 1]]
            # Calculate the mean value of the peaks
            mean_peaks = np.mean(peaks_value)
            ## feature 32-35 Energy in [0, 1/4 ]Nyf, [ 1/4 , 1/2]Nyf, [ 1/2 , 3/4 ]Nyf, [ 3/4 ,1]Nyf
            # Define the frequency ranges (normalized frequencies)
            freq_range_1 = [0, 1/8]  # Range [0, 1/4] Nyquist frequency
            freq_range_2 = [1/8, 1/4]  # Range [1/4, 1/2] Nyquist frequency
            freq_range_3 = [1/4, 3/8]  # Range [1/2, 3/4] Nyquist frequency
            freq_range_4 = [3/8, 1/2]  # Range [3/4, 1] Nyquist frequency
            energy_0_1_4 = []
            energy_1_4_1_2 = []
            energy_1_2_3_4 = []
            energy_3_4_1 = []
            # Calculate the energy in the frequency ranges
            energy_0_1_4 = np.sum(np.square(magnitude_spectrum[(freq_range_1[0] * len(magnitude_spectrum)):np.int(freq_range_1[1] * len(magnitude_spectrum))]))
            energy_1_4_1_2 = np.sum(np.square(magnitude_spectrum[np.int(freq_range_2[0] * len(magnitude_spectrum)):np.int(freq_range_2[1] * len(magnitude_spectrum))]))
            energy_1_2_3_4 = np.sum(np.square(magnitude_spectrum[np.int(freq_range_3[0] * len(magnitude_spectrum)):np.int(freq_range_3[1] * len(magnitude_spectrum))]))
            energy_3_4_1 = np.sum(np.square(magnitude_spectrum[np.int(freq_range_4[0] * len(magnitude_spectrum)):np.int(freq_range_4[1] * len(magnitude_spectrum))]))
            ## feature 36 Spectral centroid
            spectral_centroid = np.sum(freq_values * magnitude_spectrum) / np.sum(magnitude_spectrum)
            ## feature 37 Gyration radius
            mean_dft = np.mean(fft_values)
            squared_distances = np.abs(fft_values - mean_dft)**2
            mean_squared_distance = np.mean(squared_distances)
            gyration_radius = np.sqrt(mean_squared_distance)
            ## feature 38 Spectral centroid width
            centroid_width = np.sum((freq_values - spectral_centroid)**2 * magnitude_spectrum) / np.sum(magnitude_spectrum)
            
            ## feature 39  Kurtosis of the maximum of all discrete Fourier transforms (DFTs) Kurtosis as a function of time t
            Kurtosis_spec_function_t = kurtosis(np.max(spectrogram, axis=1))
            ## feature 40  Kurtosis of the maximum of all discrete Fourier transforms (DFTs) Kurtosis as a function of time f
            Kurtosis_spec_function_f = kurtosis(np.max(spectrogram, axis=0))
            ## feature 41 Mean ratio between the maximum and the mean of all DFTs
            Mean_ratio_max_mean = np.mean(np.max(spectrogram) / np.mean(spectrogram))
            ## feature 42 Mean ratio between the maximum and the median of all DFTs
            Mean_ratio_max_median = np.mean(np.max(spectrogram) / np.median(spectrogram))
            ## feature 43 Number of peaks in the curve showing the temporal evolution of the DFTs maximum
            temporal_evolution_DFT_max = np.max(spectrogram, axis=1)
            temporal_evolution_DFT_max_peaks, _ = find_peaks(temporal_evolution_DFT_max)
            # Count the number of peaks
            no_peaks_DFT_max = len(temporal_evolution_DFT_max_peaks)
            ## feature 44 Number of peaks in the curve showing the temporal evolution of the DFTs mean
            temporal_evolution_DFT_mean = np.mean(spectrogram, axis=1)
            temporal_evolution_DFT_mean_peaks, _ = find_peaks(temporal_evolution_DFT_mean)
            # Count the number of peaks
            no_peaks_DFT_mean = len(temporal_evolution_DFT_mean_peaks)
            ## feature 45 Number of peaks in the curve showing the temporal evolution of the DFTs median
            temporal_evolution_DFT_median = np.median(spectrogram, axis=1)
            temporal_evolution_DFT_median_peaks, _ = find_peaks(temporal_evolution_DFT_median)
            # Count the number of peaks
            no_peaks_DFT_median = len(temporal_evolution_DFT_median_peaks)
            ## feature 46 Ratio between 43 and 44
            if no_peaks_DFT_mean==0:
                ratio_43_44 = 0
            else:
                ratio_43_44 = no_peaks_DFT_max / no_peaks_DFT_mean
            
            ## feature 47 Ratio between 43 and 45
            if no_peaks_DFT_median==0:
                ratio_43_45 = 0
            else:
                ratio_43_45 = no_peaks_DFT_max / no_peaks_DFT_median
            
            ## feautre 48 Number of peaks in the curve of the temporal evolution of the DFTs central frequency
            fre_index, time_index = spectrogram.shape
            central_fre = np.zeros(time_index)
            max_fre = []
            frequency_mean = []
            frequency_median = []
            a_spectrum = np.copy(spectrogram)

            for i_time in range(time_index):
                a_spectrum[np.abs(spectrogram[:, i_time]) < 0.05 * np.max(np.abs(spectrogram[:, i_time])), i_time] = 0
                frequency_index_row = np.nonzero(np.abs(a_spectrum[:, i_time]) != 0)
                frequency_index_row = frequency_index_row[0]
                if frequency_index_row.size == 0:
                    cut_f1 = 1
                    cut_f2 = 120
                    frequency_mean.append(60)
                    frequency_median.append(60)
                else:
                    cut_f1 = freq[np.min(frequency_index_row)]
                    cut_f2 = freq[np.max(frequency_index_row)]
                    frequency_mean.append(np.mean(freq[frequency_index_row]))
                    frequency_median.append(np.median(freq[frequency_index_row]))
                
                central_fre[i_time] = (cut_f1 + cut_f2) / 2
                max_fre.append(cut_f2)

            peak_row, _ = find_peaks(central_fre)
            num_peak_spec_central_f = len(peak_row)
            num_peak_spec_central_fre = num_peak_spec_central_f
            ## feautre 49 Number of peaks in the curve of the temporal evolution of the DFTs max frequency
            peak_row_max, _ = find_peaks(max_fre)
            num_peak_spec_max_f = len(peak_row_max)
            ## feautre 50 Ratio between 48 and 49
            
            if num_peak_spec_max_f==0:
                ratio_48_49 = 0
            else:
                ratio_48_49 = num_peak_spec_central_fre / num_peak_spec_max_f
                
            ## feautre 51 Mean distance between the curves of the temporal evolution of the DFTs maximum frequency and mean frequency
            mean_dis_max_mean_fre = np.mean(np.array(frequency_mean) - np.array(max_fre))
            ## feautre 52 Mean distance between the curves of the temporal evolution of the DFTs maximum frequency and median frequency
            mean_dis_max_median_fre = np.mean(np.array(frequency_median) - np.array(max_fre))
            ## feautre 53 Mean distance between the 1st quartile and the median of all DFTs as a function of time
            first_quartile_index = int(np.floor(spectrogram.shape[1]) / 4)
            second_quartile_index = int(2 * np.floor(spectrogram.shape[1] / 4))
            third_quartile_index = int(3 * np.floor(spectrogram.shape[1] / 4))
            time_index_to_the_end = np.arange(spectrogram.shape[1])
            time_index_median = int(np.floor(np.median(time_index_to_the_end)))
            mean_distance_first_median = np.mean(np.abs(np.abs(spectrogram[:, first_quartile_index]) - np.abs(spectrogram[:, time_index_median])))
            ## feautre 54 Mean distance between the 3rd quartile and the median of all DFTs as a function of time
            mean_distance_third_median = np.mean(np.abs(np.abs(spectrogram[:, third_quartile_index]) - np.abs(spectrogram[:, time_index_median])))
            ## feautre 55 Mean distance between the 3rd quartile and the 1st quartile of all DFTs as a function of time
            mean_distance_first_third = np.mean(np.abs(np.abs(spectrogram[:, first_quartile_index]) - np.abs(spectrogram[:, third_quartile_index])))
            
            elapsed_time = time.time() - start_time
            # print(f"Elapsed time: {elapsed_time} seconds")

            feature_array[i_event, j_channel, 0:55] = [ratio_mean_max_env, ratio_median_max_env, kurtosis_signal, kurtosis_env, Skewness_signal, 
                                                        Skewness_env, No_auto_peaks, energy_first_3rd, energy_remaining, ratio_f8_f9, 
                                                        energy1_6hz, energy6_12hz, energy10_14hz, energy12_18hz, energy16_20hz, 
                                                        Kurtosis1_6hz, Kurtosis6_12hz, Kurtosis10_14hz, Kurtosis12_18hz, Kurtosis16_20hz,
                                                        max_env, mean_DFT, max_DFT, max_frequency, centroid, 
                                                        first_quartile_frequency, second_quartile_frequency, median_normalized, Var_normalized, num_peaks_75_DFT,
                                                        mean_peaks, energy_0_1_4, energy_1_4_1_2, energy_1_2_3_4, energy_3_4_1,
                                                        spectral_centroid, gyration_radius, centroid_width, Kurtosis_spec_function_t, Kurtosis_spec_function_f,
                                                        Mean_ratio_max_mean, Mean_ratio_max_median, no_peaks_DFT_max, no_peaks_DFT_mean, no_peaks_DFT_median,
                                                        ratio_43_44, ratio_43_45, num_peak_spec_central_fre, num_peak_spec_max_f, ratio_48_49,
                                                        mean_dis_max_mean_fre, mean_dis_max_median_fre, mean_distance_first_median, mean_distance_third_median, mean_distance_first_third]

                                                        
    return feature_array


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

def plotter_stft(x_stft):
    fig, axs = plt.subplots(6,4,figsize=(25,25))
    fig.suptitle('Seismic signal', fontsize=24)
    for i in range(6):
        axs[i, 0].pcolormesh(np.abs((x_stft[0,i,:,:])))
        axs[i, 0].set_xticks(np.arange(0, 101, 50))
        axs[i, 0].set_xticklabels([0, 5, 10])
        axs[i, 0].set_yticks(np.arange(0, 129, 64))
        axs[i, 0].set_yticklabels([0, 62.5, 125])
        axs[0, 0].set_title('Rockfall', fontsize=20)
        axs[i, 0].set_ylabel('Frequency (Hz)')
        if i==5:
            axs[i, 0].set_xlabel('Time (s)')

        axs[i, 1].pcolormesh(np.abs((x_stft[500,i,:,:])))
        axs[0, 1].set_title('Quake', fontsize=20)
        axs[i, 1].set_xticks(np.arange(0, 101, 50))
        axs[i, 1].set_xticklabels([0, 5, 10])
        axs[i, 1].set_yticks(np.arange(0, 129, 64))
        axs[i, 1].set_yticklabels([0, 62.5, 125])
        axs[i, 1].set_ylabel('Frequency (Hz)')
        if i==5:
            axs[i, 1].set_xlabel('Time (s)')

        axs[i, 2].pcolormesh(np.abs((x_stft[700,i,:,:])))
        axs[0, 2].set_title('Earthquake', fontsize=20)
        axs[i, 2].set_xticks(np.arange(0, 101, 50))
        axs[i, 2].set_xticklabels([0, 5, 10])
        axs[i, 2].set_yticks(np.arange(0, 129, 64))
        axs[i, 2].set_yticklabels([0, 62.5, 125])
        axs[i, 2].set_ylabel('Frequency (Hz)')
        if i==5:
            axs[i, 2].set_xlabel('Time (s)')

        axs[i, 3].pcolormesh(np.abs((x_stft[1100,i,:,:])))
        axs[0, 3].set_title('Noise', fontsize=20)
        axs[i, 3].set_xticks(np.arange(0, 101, 50))
        axs[i, 3].set_xticklabels([0, 5, 10])
        axs[i, 3].set_yticks(np.arange(0, 129, 64))
        axs[i, 3].set_yticklabels([0, 62.5, 125])
        axs[i, 3].set_ylabel('Frequency (Hz)')
        if i==5:
            axs[i, 3].set_xlabel('Time (s)')

    plt.show()

def plotter_time_series(x):
    fig, axs = plt.subplots(6,4,figsize=(25,25))
    fig.suptitle('Seismic signal', fontsize=24)
    for i in range(6):
        axs[i, 0].plot((x[0,:, i]))
        axs[i, 0].set_xticks(np.arange(0, 2500, 1249))
        axs[i, 0].set_xticklabels([0, 5, 10])
        axs[i, 0].set_yticks(np.arange(-1, 1, 0.99))
        axs[i, 0].set_yticklabels([-1, 0, 1])
        axs[0, 0].set_title('Rockfall', fontsize=20)
        axs[i, 0].set_ylabel('Amplitude')
        if i==5:
            axs[i, 0].set_xlabel('Time (s)')

        axs[i, 1].plot((x[500,:, i]))
        axs[i, 1].set_xticks(np.arange(0, 2500, 1249))
        axs[i, 1].set_xticklabels([0, 5, 10])
        axs[i, 1].set_yticks(np.arange(-1, 1, 0.99))
        axs[i, 1].set_yticklabels([-1, 0, 1])
        axs[0, 1].set_title('Quake', fontsize=20)
        axs[i, 1].set_ylabel('Amplitude')
        if i==5:
            axs[i, 1].set_xlabel('Time (s)')

        axs[i, 2].plot((x[700,:, i]))
        axs[i, 2].set_xticks(np.arange(0, 2500, 1249))
        axs[i, 2].set_xticklabels([0, 5, 10])
        axs[i, 2].set_yticks(np.arange(-1, 1, 0.99))
        axs[i, 2].set_yticklabels([-1, 0, 1])
        axs[0, 2].set_title('Earthquake', fontsize=20)
        axs[i, 2].set_ylabel('Amplitude')
        if i==5:
            axs[i, 2].set_xlabel('Time (s)')

        axs[i, 3].plot((x[1100,:, i]))
        axs[i, 3].set_xticks(np.arange(0, 2500, 1249))
        axs[i, 3].set_xticklabels([0, 5, 10])
        axs[i, 3].set_yticks(np.arange(-1, 1, 0.99))
        axs[i, 3].set_yticklabels([-1, 0, 1])
        axs[0, 3].set_title('Noise', fontsize=20)
        axs[i, 3].set_ylabel('Amplitude')
        if i==5:
            axs[i, 3].set_xlabel('Time (s)')

    plt.show()

def k_means_cluster(data, true_label, n_clusters=4):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=123, n_init=20)
    y_pred = kmeans_model.fit_predict(data) ## reshape data to 2 dim
    y_pred = np.copy(y_pred)
    y_true = np.argmax(true_label, axis=1)
    print('k-means acc = %f', np.round(acc(y_true, y_pred), 5))
    label = y_pred
    center = kmeans_model.cluster_centers_
    acc_value = np.round(acc(y_true, y_pred), 5)
    return label, center, acc_value

def Gaussian_mixture_model_cluster(data, true_label, n_clusters=4):
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(data) ## data need to be reshape to 2 dim
    gmm_labels = gmm.predict(data)
    y_true = np.argmax(true_label, axis=1)
    print('GMM acc = %f', np.round(acc(y_true, gmm_labels), 5))
    label = gmm_labels
    acc_value = np.round(acc(y_true, gmm_labels), 5)
    return label, acc_value


def PCA_com(data):
    pca = PCA(n_components=1000)  # Specify the number of components you want to keep

    # Fit the PCA model to your data
    pca.fit(data)  # X is your data matrix or array

    # Transform the data to the principal components
    PCA_coeff = pca.transform(data)

    return PCA_coeff


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[ind[0][i], ind[1][i]] for i in range(D)]) * 1.0 / y_pred.shape[0]