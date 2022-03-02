'''
This script implements the LMS based Active Noise Cancellation algorithm.
We have access to the source signal. The algorithm actively predicts the Wiener Filter weights
to compute the LMS estimate of the source signal from the noisy signal.
'''

import numpy as np
import scipy as sc
from scipy.io import wavfile
import src.utils as util


folder_path = 'data/MS-SNSD/clean_test/'
file_path = util.random_file_gen(folder_path = folder_path, num_files = 2)

Fs, audio_dat = wavfile.read(file_path[0])

time = np.linspace(0, len(audio_dat)/Fs, len(audio_dat))

sir = -20
snr = -20

interere = 1 # Binary value 1 if need interferer, 0 if not.

if interere:
    _, interfere_dat = wavfile.read(file_path[1])
    interfere_dat = interfere_dat/np.max(np.abs(interfere_dat))
    interfere_dat = util.interferer(audio_dat, interfere_dat, sir)
else:
    interfere_dat = np.zeros(shape = time.shape)

noise = util.white_noise(audio_dat, snr)

sensor_sig = audio_dat + noise + interfere_dat
print(sensor_sig.shape, time.shape)

M = 16
mu = 1e-2
w = np.zeros(shape = (M), dtype = np.complex64)
y = np.zeros(shape = time.shape, dtype = np.complex64)
e = y

for m in range(M, len(time)):
    s_sum = 0
    for i in range(0, M):
        s_sum = s_sum + w[i]*sensor_sig[m - i]
    
    y[m] = s_sum
    e[m] = audio_dat[m] - y[m]
    
    for i in range(0, M):
        w[i] = w[i] + 2*mu*e[m]*sensor_sig[m - i]

