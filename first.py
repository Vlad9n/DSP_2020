import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from scipy.io import wavfile


logging.basicConfig(format="[%(filename)s: %(funcName)s] %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
FILE_PATH = r"/home/warlock/projects/dsp_lab/lab_0/voice.wav"

def dz1():
    F = 10
    samplerate = 1000
    ampl = np.iinfo(np.int16).max // 2
    logger.info(f"amplitude = {ampl}")
    t = np.linspace(0, 1, samplerate)
    S = ampl * np.sin(2 * np.pi * F * t)
    noise = ampl * (np.random.rand(samplerate) * 2 - 1)
    S_round = np.round(S)
    noise_round = np.round(noise)
    S_err = S - S_round
    noise_err = noise - noise_round

    S_sn = 10 * np.log10(np.var(S)/np.var(S_err))
    noise_sn = 10 * np.log10(np.var(noise) / np.var(noise_err))
    print(f"theor = {6*16 - 7.2}, signal = {S_sn}, noise = {noise_sn}")



def gen_signal():
    samplerate = 2000
    ampl = np.iinfo(np.int16).max / 2
    t = np.linspace(0, 1, samplerate)
    F1 = 250  # frequent
    F2 = 500
    F3 = 750
    S1 = np.sin(2 * np.pi * F1 * t)
    S2 = np.sin(2 * np.pi * F2 * t)
    S3 = np.sin(2 * np.pi * F3 * t)
    plt.figure()
    plt.plot(t, S1, '-r', label='Freq=250')
    plt.plot(t, S2, '-g', label='Freq=500')
    plt.plot(t, S3, '-b', label='Freq=750')
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.legend()
    wavfile.write('250.wav', samplerate, S1.astype(np.int16))

    plt.show()

if __name__=="__main__":
    samplerate, data = wavfile.read(FILE_PATH)
    #plt.plot(range(500), data[:500], '-g')
    #plt.show()

    #data = ampl * np.sin(2. * np.pi * freq * t/samplerate + phase)
    dz1()


