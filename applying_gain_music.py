from imports import *
import user_inputs
import applying_gain_noise
import music_tf

def apply_snr(music, noise):
    modified_music = []
    full_noise = []
    for iii in range(0, len(user_inputs.SNRs)):
        for ii in range (0, len(noise)):    # includes gains already
            for i in range (0, len(music)): # includes TFs already
                my_music=music[i]
                my_noise=noise[ii]

                needed_SNR= user_inputs.SNRs[iii]

                noise_power = np.mean(my_noise**2)
                music_power = np.mean(my_music**2)
                snr_as_imported = music_power/noise_power
                factor = np.sqrt(needed_SNR/snr_as_imported)
                modified_music.append(factor*my_music)
                full_noise.append(my_noise)

    return np.array(modified_music), np.array(full_noise)

Data_I, Data_H = apply_snr(music_tf.Data_F, applying_gain_noise.Data_C)

"""
the final shape of Data_I is what we feed to the mixer, windower, and then the ML:
    for SNR
        for noise gain
            for noise file
                for music tf
                    for music file

"""

