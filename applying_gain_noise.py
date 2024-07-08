from imports import *
import user_inputs
import resampling

def adding_gain_noise(data,gains):
    desired_gains_linear = [10**(gain /20) for gain in gains]
    
    adjusted_rec = {}
    master_c = 0
    for gain in desired_gains_linear:
        for i in range (0, len(data)):
            adjusted_single_rec = data[i] * gain 
            adjusted_rec[master_c] = adjusted_single_rec
            master_c+=1
    return adjusted_rec

Data_C = adding_gain_noise(resampling.Data_B, user_inputs.noise_gains)