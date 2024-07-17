from imports import *
import user_inputs
import resampling

def adding_gain_noise(data,gains):
    desired_gains_linear = [10**(gain /20) for gain in gains]
    
    adjusted_rec = []
    for gain in desired_gains_linear:
        for i in range (0, len(data)):
            adjusted_single_rec = data[i] * gain 
            adjusted_rec.append(adjusted_single_rec)
    return adjusted_rec

Data_C = adding_gain_noise(resampling.Data_B, user_inputs.noise_gains)

"""
The shape of Data C is:
    for noise gain
        for noise file
- It's final shape should equal (the number of noise files * number of noise gains , noise length in samples)

"""
