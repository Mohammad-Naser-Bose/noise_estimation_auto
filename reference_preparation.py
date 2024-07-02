from imports import *
import applying_gain_noise
import resampling
import user_inputs


def generate_perm_music(music):
    duplicated_music = {}
    master_c = 0
    for iii in range(0, len(user_inputs.sought_ratio)):
        for ii in range (0, user_inputs.num_noise_combinations):
            for i in range (0, len(music)):
                duplicated_music[master_c]=music[i]
                master_c+=1
    return duplicated_music

Data_L = generate_perm_music(resampling.Data_E)