
import os
import argparse
import csv

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import tqdm.autonotebook as tqdm
from sklearn.preprocessing import OneHotEncoder
from k_means_mm import _k_means_minus_minus
import random
import torch
import cupy as cp
import torch.cuda
import scipy.io as sio

from utils import *
import torch.optim as optim

def kmeans_cuda(X, n_clusters, max_iterations=20, random_seed=None):
    # Set random seed
    if random_seed is not None:
        cp.random.seed(random_seed)
    
    # Initialize centroids randomly
    centroids = X[cp.random.choice(X.shape[0], n_clusters, replace=False)]
    
    for _ in range(max_iterations):
        # Calculate distances between points and centroids
        distances = cp.linalg.norm(X[:, cp.newaxis] - centroids, axis=-1)
        
        # Assign points to the nearest centroid
        labels = cp.argmin(distances, axis=-1)
        
        # Update centroids
        new_centroids = cp.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Check for convergence
        if cp.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels.get(), centroids.get()

dataset_path = ## your data path
x, y = load_time_seties_data(dataset_path)
No_samples = 
No_channels = 
Len_time_signal = 
## add noise
noise = np.random.normal(loc=0, scale=0.15, size=(No_samples,Len_time_signal,No_channels))
x = x + noise

y_onedim = np.argmax(y, axis=1)
filtered_array, f, x_stft_normalized_spectrogram = bandpass_stft_filter(x, low_frequency=1, upper_frequency=20, fs=250)
x_stft_normalized_spectrogram = np.nan_to_num(x_stft_normalized_spectrogram)

No_frequency_bin = 
No_time_bin = 

k_means_cluster(np.reshape(x_stft_normalized_spectrogram,[No_samples,No_channels*No_frequency_bin*No_time_bin]), y, n_clusters=4)
num_run = 20 # 20
num_basic_partitions = 50 #100
input_data = np.reshape(x_stft_normalized_spectrogram,[No_samples,No_channels*No_frequency_bin*No_time_bin])

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()
final_binary_matrix_B = np.zeros((No_samples, 0))
prop_outliers_list = [0.04] # 0.01, 0.02, 0.03, 0.05

for i_prop in range(len(prop_outliers_list)):
    
    for i_run in np.arange(0, num_run, 1, dtype=int): #range(num_run):
        acc_value = np.array([])
        samples_list = np.array([])
        # Set the range
        range_min = 10
        range_max = 40

        # Set the mean and standard deviation
        mean = 4
        std_dev = 20

        # Generate random numbers from the normal distribution
        samples = []
        while len(samples) < 50:
            new_samples = np.random.normal(mean, std_dev, 1)
            if range_min<=new_samples<=range_max: #= np.clip(new_samples, range_min, range_max)
                samples.extend(new_samples)

        # Truncate extra samples if generated more than 100
        samples = np.floor(samples[:50])
        samples = samples.astype(int)
        samples_list = np.concatenate([samples_list, samples])
        for i in range(num_basic_partitions):
            cluster_index = random.randint(0, 6)
            
            ## gpu acc
            # # Convert data to cupy array for GPU computations
            X_gpu = cp.array(input_data)
            labels, centroids = kmeans_cuda(X_gpu, n_clusters=samples[i], random_seed=i)

            # Convert results back to numpy arrays
            labels = cp.asnumpy(labels)
            centroids = cp.asnumpy(centroids)

            y_pred_gpu= labels
            y_pred = np.copy(y_pred_gpu)
            encoded_data = encoder.fit_transform(np.reshape(y_pred, [No_samples,1]))
            # Convert the sparse matrix to an array
            binary_matrix_B = encoded_data.toarray()
            final_binary_matrix_B = np.concatenate([final_binary_matrix_B, binary_matrix_B], axis=1)

        final_binary_matrix_B_hat = 1 - final_binary_matrix_B

        whole_B = np.concatenate([final_binary_matrix_B, final_binary_matrix_B_hat], axis=1)
        sample_weight = np.ones([No_samples]) # whole_B
        best_labels, best_inertia, best_centers,outlier_indices = _k_means_minus_minus(whole_B, sample_weight = sample_weight, n_clusters=4, prop_outliers=prop_outliers_list[i_prop], max_iter=300, init="k-means++", verbose=False, x_squared_norms=None, random_state=123, tol=1e-4, precompute_distances=True)
        
        print('k-means_MM acc = %f', np.round(acc(y_onedim, best_labels), 5))
        acc_value = np.concatenate([acc_value, np.array([np.round(acc(y_onedim, best_labels), 5)])])
        nmi = np.round(normalized_mutual_info_score(y_onedim, best_labels), 5)
        
        sio.savemat('best_centers.mat', {'data': best_centers})
        
        # Convert acc_value to a list or array
        acc_value_list = [acc_value]    # List containing acc_value
    
        # Construct the file path
        file_path = 'conmuse_k_i_accwithprop_outliers_{}_run_{}.csv'.format(str(prop_outliers_list[i_prop]), str(i_run))
    
        # Save data to CSV
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([acc_value_list])  # Pass acc_value_list as the iterable
        acc_value_list = []
         
        # Convert acc_value to a list or array
        sample_value_list = [samples_list]      # List containing acc_value
    
        # Construct the file path
        file_path = 'conmuse_k_i_accwithprop_samples_{}_run_{}.csv'.format(str(prop_outliers_list[i_prop]), str(i_run))
    
        # Save data to CSV
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([sample_value_list])  # Pass acc_value_list as the iterable
        sample_value_list = []

        # Convert acc_value to a list or array
        outlier_indices_list = [outlier_indices]      # List containing acc_value

        # Construct the file path
        file_path = 'conmuse_k_i_outlier_indices_withprop_samples_{}_run_{}.csv'.format(str(prop_outliers_list[i_prop]), str(i_run))
    
         # Save data to CSV
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([outlier_indices_list])  # Pass acc_value_list as the iterable
        outlier_indices_list = []
        
        # Convert acc_value to a list or array
        nmi_list = [nmi]      # List containing acc_value
        
        # Construct the file path
        file_path = 'conmuse_k_i_nmiwithprop_samples_{}_run_{}.csv'.format(str(prop_outliers_list[i_prop]), str(i_run))
    
        # Save data to CSV
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([nmi_list])  # Pass acc_value_list as the iterable
        nmi_list = []
        
        # Convert acc_value to a list or array
        best_labels_list = [best_labels]      # List containing acc_value
        
        # Construct the file path
        file_path = 'conmuse_k_i_best_labelswithprop_samples_{}_run_{}.csv'.format(str(prop_outliers_list[i_prop]), str(i_run))
    
        # Save data to CSV
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([best_labels_list])  # Pass acc_value_list as the iterable
        best_labels_list = []
