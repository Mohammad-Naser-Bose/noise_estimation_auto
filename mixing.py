from imports import *
import applying_gain_music
from decimal import Decimal, getcontext
getcontext().prec = 15


def concatenating_noise(music, noise):
    noise_data_full_length_dic = []
    for i in range (0, len(music)):
        repeat_factor = int(np.ceil(len(music[i]) / len(noise[i])))
        noise_data_full_length = np.tile(noise[i], repeat_factor)[:len(music[i])]
        noise_data_full_length_dic.append(noise_data_full_length)
    return np.array(noise_data_full_length_dic)
def mixing(audio, noise):
    mixed_signal_sin_rec = []
    for i in range (0, len(audio)):
        mixed_signal_sin_rec.append(audio[i] + noise[i])
    return np.array(mixed_signal_sin_rec)


Data_J = concatenating_noise(applying_gain_music.Data_I, applying_gain_music.Data_H)
Data_K = mixing(applying_gain_music.Data_I, Data_J)

stop=1