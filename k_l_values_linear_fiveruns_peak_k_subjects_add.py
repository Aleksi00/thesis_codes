# Imports:
import sys
import time
import numpy as np
import pickle
import cupy as cp
import pandas as pd


import os
import os.path as op
from kuramoto_functions import dfa_fitting, tune_sc_percentile, simulate_single_run
from cross_analytics.crosspy.core import criticality
from cross_analytics.crosspy.core import synchrony
#from crosspy.core import criticality
#from crosspy.core import synchrony

from utils.pac import compute_pac_spectrum, _pac_kernel, compute_pac_with_lags
from utils.pac import transform_to_cdf, get_length_by_cdf

import scipy as sp
import logging
logging.basicConfig(filename='logs.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
logging.info("let's get it started")

### KURAMOTO PARAMETERS:
n_nodes = 200
num_oscillators = 1000
sr = 250
f_nodes = [10] * n_nodes
f_spread = 1.0
noise_sigma = 3.0
time_s = 300
window_sizes = np.geomspace(20 / 10 * sr, (time_s - 50) * sr * 0.2, 30).astype(int)

lags_cycles = np.arange(0, 20, 0.1)
freqs =  [10] * n_nodes #np.load("./data/freqs.npy") # np.geomspace(1, 80, 51)
sr = 250

#clinical_ids = np.load('clinical_ids_all_add.npy')[100:150]
clinical_ids = np.load('/m/nbe/scratch/leap_mcpsych/Samanta/kuramoto/clinical_ids_all_add.npy')[100:150]


ls = [5] #ls = np.linspace(1, 10, num=40)  #dfa as a funciton of L
exps = 2/3 #[1]

ks = np.linspace(1, 8, num=40)  # Or load from wherever you get them; #noneven_space(8, 18, 20)#int(sys.argv[1])

s_i = int(sys.argv[1])
n_subjs = 50

subj_idx, l_idx, k_idx = np.unravel_index(s_i, ( n_subjs, len(ls), len(ks))) #subj_idx, l_idx, k_idx = np.unravel_index(s_i, ( n_subjs, len(ls),  len(k)))

c_id = clinical_ids[subj_idx]

k_path = f"/m/nbe/scratch/leap_mcpsych/Samanta/kuramoto/brainplots/peak_k_values_all_newest/peak_ks_subject_{c_id}.csv"
k_shape = pd.read_csv(k_path, sep=',', header=0)
k_shape = k_shape.values.flatten()
print(k_shape.shape)
mean_k = np.mean(k_shape)
k = k_shape / mean_k # Mean normalize k
print(k.shape)

# Mean of peak k values 
# k = [k]
# k_path = f"/m/nbe/scratch/leap_mcpsych/Samanta/kuramoto/brainplots/k_values/peak_ks_subject_19.csv"
# k_shape = pd.read_csv(k_path, sep=',', header=0)
# k = k_shape.values.flatten()
# print(k.shape)
# mean_k = np.mean(k)
# k = [mean_k]

l = ls[l_idx]
k = k * ks[k_idx] # k = k[k_idx] ; # k = ks[k_idx]

subject_path = f"/m/nbe/scratch/leap_mcpsych/derivatives/DTI/ses-01/kuramoto/clinical/all_subjects/sub-CON{c_id}_parcels_coreg_yeo17_200.csv"
subject = pd.read_csv(subject_path, sep=',', header=None)
print(np.array(subject).shape)
#orig_sc = subject
orig_sc = (np.array(subject)[:200, :200])
connectome_normed = orig_sc / orig_sc.mean()
print(np.array(connectome_normed).shape)

# subject = pickle.load(open(subject_path, 'rb'))
# subject = pd.read_csv(subject_path, sep=' ', header=None)
# subject = pd.read_csv(subject_path, header=None, delimiter=",")
# orig_sc = (np.array(subject)[:200, :200])
# print(np.array(orig_sc).shape)
# orig_sc = subject["connectome"] ** exps #/ np.mean(subject["connectome"])
# orig_sc = subject ** exps #/ np.mean(subject["connectome"])
# orig_sc = subject
# connectome_normed = subject

logging.info(f"sub-{c_id}")

sim = f"sub-{c_id}"

save_path = op.join("/m/nbe/scratch/leap_mcpsych/Samanta/kuramoto/different_subjects/k_l/subjects/clinical_peak_k_newest/", sim)
os.makedirs(save_path, exist_ok=True)

#file_name_body = "log_K-" + str(k) + "_L-" + str(l)
file_name_body = "log_K-" + str(ks[k_idx]) + "_L-" + str(l)
timeseries_file_name = file_name_body + f"_ts.npy"
timeseries_file_path = f'{save_path}/{timeseries_file_name}'
for seed in range(5):
    
    if not(os.path.exists(timeseries_file_path)):
        results = simulate_single_run(k, node_frequencies=f_nodes, time=time_s, frequency_spread=f_spread, aggregate='mean',
                                        n_oscillators=num_oscillators, weight_matrix=l * connectome_normed, sr=sr, use_tqdm=False, noise_sigma=noise_sigma, omegas=None, omega_seed=seed, random_seed=seed)
        logging.info("simulated")


        # Save the timeseries:
        timeseries_file_name = file_name_body + f"_ts_seed{seed}.npy"
        np.save(f"{save_path}/{timeseries_file_name}", results)
        results = results[:, sr*50:]

        data_gpu = cp.real(results)
        pac_spectrum = compute_pac_spectrum(data_gpu, frequencies=freqs, sampling_rate=sr, lags_cycles=lags_cycles)
        pacf_lifetime = pac_spectrum[:, :, 10:50].mean(axis=-1)


        np.save(f"{save_path}/pac_spectrum-{file_name_body}_fr{f_spread}_ex{exps}_seed{seed}.npy", pac_spectrum)
        np.save(f"{save_path}/pacf_lifetime-{file_name_body}_fr{f_spread}_ex{exps}_seed{seed}.npy", pacf_lifetime)
        #         np.save(f"{save_path}/K-{file_name_body}_fr{f_spread}_ex{exps}.npy", k * k_is)

        # Calculate PLV and DFA:
        sim_envelope = np.abs(results)
        model_dfa = criticality.dfa(sim_envelope, window_sizes)[2]
        cplv = synchrony.cplv(results)
        wpli = np.abs(synchrony.wpli(results))
        plv = np.abs(cplv)
        iplv = np.imag(cplv)


        results_gpu = cp.asarray(sim_envelope)
        correlation_matrix = cp.corrcoef(results_gpu)

        # Save results to the recently created folder:
        occ_file_name = file_name_body + f"_fr{f_spread}_ex{exps}_occ_seed{seed}.npy"
        np.save(op.join(save_path, occ_file_name), correlation_matrix.get())

        # Save results to the recently created folder:
        plv_file_name = file_name_body + f"_fr{f_spread}_ex{exps}_plv_seed{seed}.npy"
        np.save(op.join(save_path, plv_file_name), plv)

        iplv_file_name = file_name_body + f"_fr{f_spread}_ex{exps}_iplv_seed{seed}.npy"
        np.save(op.join(save_path, iplv_file_name), iplv)

        wpli_file_name = file_name_body + f"_fr{f_spread}_ex{exps}_wpli_seed{seed}.npy"
        np.save(op.join(save_path, wpli_file_name), wpli)

        dfa_file_name = file_name_body + f"_fr{f_spread}_ex{exps}_dfa_seed{seed}.npy"
        np.save(op.join(save_path, dfa_file_name), model_dfa)

        order_file_name = file_name_body + f"_fr{f_spread}_ex{exps}_order_seed{seed}.npy"
        np.save(op.join(save_path, order_file_name),  np.mean(sim_envelope, axis=1))

