from imports import *
import user_inputs
import applying_gain_noise
import permutation_noise_music
import music_tf





def find_RMS_noise(data):
    RMS_values = []
    for i, recording in enumerate (data.items()):
        my_data = recording[1]
        RMS_values.append(librosa.feature.rms(y=np.array(my_data)).mean())
        RMS_values_avg = np.mean(RMS_values)
    return RMS_values_avg

def find_RMS_data(data):
    RMS_values = []
    for i, recording in enumerate (data.items()):
        my_data = recording[1]
        RMS_values.append(librosa.feature.rms(y=np.array(my_data)).mean())
        RMS_values_avg = np.mean(RMS_values)
    return RMS_values_avg


def apply_music_gain(noise, audio_full, audio_short):
    noise_rms_avg = find_RMS_noise(noise)
    data_rms_avg = find_RMS_data(audio_short)

    current_rms_ratio = noise_rms_avg/data_rms_avg

    modified_factors = []
    for val in user_inputs.sought_ratio:
        modified_factors.append(current_rms_ratio * val)

    length_seg = int(len(audio_full)/len(user_inputs.sought_ratio))
    modified_factors_dict = {}
    i=0
    beg_index = i
    for mod_f in modified_factors:
        end_index = i+length_seg
        for index in range(beg_index, end_index):
            modified_factors_dict[i] = mod_f
            i+=1
        beg_index=end_index


    modified_music = {}
    for key, value in audio_full.items():
        modified_music[key] = audio_full[key]/ modified_factors_dict[key] 
    return modified_music


Data_I = apply_music_gain(applying_gain_noise.Data_C, permutation_noise_music.Data_G,music_tf.Data_F)
