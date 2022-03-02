'''
This script implements the LMS based Active Noise Cancellation algorithm.
We have access to the source signal. The algorithm actively predicts the Wiener Filter weights
to compute the LMS estimate of the source signal from the noisy signal.
'''

import numpy as np
import scipy as sc
from scipy.io import wavfile
import src.utils as util
import matplotlib.pyplot as plt

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

M = 16
mu = 1e-2
w = np.zeros(shape = (M), dtype = np.complex64)
y = np.zeros(shape = time.shape, dtype = np.complex64)
e = y


for m in range(M, len(time)):
    s_sum = 0
    ind_forward = np.linspace(0, M - 1, M).astype(int)
    ind_reverse = M - np.linspace(0, M - 1, M).astype(int)

    s_sum = s_sum + np.inner(w, sensor_sig[ind_reverse])
    y[m] = s_sum
    e[m] = audio_dat[m] - y[m]    

    w = w + 2*mu*np.inner(e[ind_forward], sensor_sig[ind_reverse])

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(time, np.real(audio_dat))
plt.title('Original Signal')

plt.subplot(3, 1, 2)
plt.plot(time, np.real(sensor_sig))
plt.title('Collected Signal')

plt.subplot(3, 1, 3)
plt.plot(time, np.real(e))
plt.plot(time, np.real(audio_dat - y*np.exp(-1j*M/len(time)*np.pi)))
plt.legend(['Prediction Error', 'Phase shifted predicted error'])
plt.title('Prediction Error')  
plt.show() 