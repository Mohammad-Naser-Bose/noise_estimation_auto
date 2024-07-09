from imports import *
import resampling
import user_inputs
import applying_gain_noise

def generate_perm_music(music,noise):
    duplicated_music = {}
    master_c = 0
    for iii in range(0, len(user_inputs.sought_ratio)):
        for ii in range (0, len(noise)):
            for i in range (0, len(music)):
                duplicated_music[master_c]=music[i]
                master_c+=1
    return duplicated_music

Data_L = generate_perm_music(resampling.Data_E, applying_gain_noise.Data_C)