'''
This script implements the LMS based Active Noise Cancellation algorithm.
We have access to the source signal. The algorithm actively predicts the Wiener Filter weights
to compute the LMS estimate of the source signal from the noisy signal.
'''

import numpy as np
import scipy as sc
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
import os
from src.utils import random_file_gen


folder_path = 'data/MS-SNSD/clean_test/'
file_path = random_file_gen(folder_path = folder_path, num_files = 2)

Fs, audio_dat = wavfile.read(file_path[0])

time = np.linspace(0, len(audio_dat)/Fs, len(audio_dat))

interere = 1 # Binary value if I need the interferer or not

if interere:
    _, interfere_dat = wavfile.read(file_path[1])
    interfere_dat = interfere_dat/np.max(np.abs(interfere_dat))
else:
    interfere_dat = np.zeros(shape = time.shape)