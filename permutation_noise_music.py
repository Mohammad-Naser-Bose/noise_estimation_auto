from imports import *
import user_inputs
import music_tf
import applying_gain_noise

def permutations(music, noise):
    duplicated_music = {}
    duplicated_noise = {}
    master_c = 0
    for iii in range(0, len(user_inputs.sought_ratio)):
        for ii in range (0, len(noise)):
            for i in range (0, len(music)):
                duplicated_music[master_c]=music[i]
                duplicated_noise[master_c]=noise[ii]
                master_c+=1
    return duplicated_music, duplicated_noise

Data_G, Data_H = permutations(music_tf.Data_F, applying_gain_noise.Data_C)

stop=1