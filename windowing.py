
from imports import *
import mixing
import reference_preparation
import user_inputs
import pickle
import applying_gain_music
from decimal import Decimal, getcontext
getcontext().prec = 15


def windowing(signal):
    master_c = 0
    windowed_data = []
    for i, rec in enumerate (signal.items()):
        my_rec = rec[1]
        num_windows = len(my_rec) // user_inputs.window_len_sample_downsampled
        extra_samples = len(my_rec) % (num_windows*user_inputs.window_len_sample_downsampled)
        truncated_rec = my_rec[extra_samples:]
        rec_windows = np.array_split(truncated_rec, num_windows)
        rec_windows_truncated = [window[:user_inputs.window_len_sample_downsampled] for window in rec_windows]
        for arr in rec_windows_truncated:
            windowed_data.append(arr.tolist())
            master_c+=1
    return np.array(windowed_data)


Data_Y=windowing(reference_preparation.Data_L)
Data_X=windowing(mixing.Data_K)
Data_O=windowing(mixing.Data_J)




# noise_power_all =[]
# all_noise=[np.array(value) for value in mixing.Data_J.values()]
# noise = np.stack(all_noise,axis=0)
# for a in noise:
#     noise_power_all.append(np.mean(a**2))



# a_power_all =[]
# aaa=[np.array(value) for value in mixing.Data_K.values()]
# bbb=[np.array(value) for value in reference_preparation.Data_L.values()]
# aaaarr = np.stack(aaa,axis=0)
# bbbarr = np.stack(bbb,axis=0)
# x=aaaarr-bbbarr
# for a in x:
#     a_power_all.append(np.mean(a**2))

stop=1