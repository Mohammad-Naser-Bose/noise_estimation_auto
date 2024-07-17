from imports import *
import resampling
import user_inputs
import applying_gain_noise

def generate_perm_music(music,noise):
    duplicated_music = []
    for iiii in range(0, len(user_inputs.SNRs)):
        for iii in range (0, len(noise)):
            for ii in range (0, user_inputs.num_tfs):
                for i in range (0, len(music)):
                    duplicated_music.append(music[i])
    return np.array(duplicated_music)


Data_L = generate_perm_music(resampling.Data_E, applying_gain_noise.Data_C)


"""
- We are taking the music from the sampling function without any modification, 
that's why we have another loop to account for the number of TFs needed (if None, TF=1)
- The final shape is always (num of SNRs * num of noise files (including gains) * num of TFs * num of music files , length of music taken)
- It's always chuncks of the original time files repeating every n number of music files
"""