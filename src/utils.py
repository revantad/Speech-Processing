from re import S
import numpy as np
import scipy as sc
import os

def random_file_gen(folder_path, num_files = 1):
    files = os.listdir(folder_path)
    file_path = np.empty(shape = num_files)
    for i in range(0, num_files):
        rand = np.random.randint(len(files))
        file_path[i] = folder_path + files[rand]

    return file_path    

def white_noise(signal, snr):
    sig_len = len(signal)
    