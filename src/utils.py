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
    sensor_rms_amp = np.sqrt(np.mean(np.abs(signal)**2))
    noise_amp = sensor_rms_amp/(10**(snr/20))
    noise = noise_amp*np.random.normal(0, 1, sig_len)
    return noise

def interferer(signal, interfere_signal, sir):
    sensor_rms_amp = np.sqrt(np.mean(np.abs(signal)**2))
    interferer_amp = sensor_rms_amp/(10**(sir/20))

    interfere = np.zeros(signal.shape, dtype = np.complex64)
    noise_rms = np.sqrt(np.mean(np.abs(signal)**2))
    
    if(len(interfere_signal)>=len(signal)):
        interfere = 1.414*interferer_amp/noise_rms*interfere_signal[0:len(signal)]
    else:
        interfere[0:len(interfere_signal)] = 1.414*interferer_amp/noise_rms*interfere_signal
    
    return interfere
    
def gccphat(Refsig, Sig, fs, d, interp = 16, c = 343):
    max_tau = None
    sig1 = Refsig
    sig2 = Sig
    m_len = len(sig1) + len(sig2)

    sig = sc.fft(sig2, n = m_len)
    refsig = sc.fft(sig1, n = m_len)

    R = sig*np.conj(refsig)

    cc = sc.ifft(R/np.abs(R), n = interp*m_len)
    max_shift = int(interp*m_len/2)

    if max_tau:
        max_shift = np.minimum(int(interp*fs*max_tau), max_shift)
    
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift/float(interp*fs)
    aoa = np.arcsin(c*tau/d) # Angle of arrival
    
    return aoa, tau, cc