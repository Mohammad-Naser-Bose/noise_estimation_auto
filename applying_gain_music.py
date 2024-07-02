from imports import *
import user_inputs
import applying_gain_noise
import permutation_noise_music





def find_RMS_noise(data):
    RMS_values = {}
    for i, recording in enumerate (data.items()):
        my_data = recording[1]
        RMS_values [i] =  librosa.feature.rms(y=np.array(my_data)).mean()
    
    RMS_values_new = {}
    for key, value in RMS_values.items():
        RMS_values_new[key] = np.array(value, dtype= np.float32)
    return RMS_values_new

def find_RMS_data(data):
    RMS_values = {}
    for i, recording in enumerate (data.items()):
        my_data = recording[1]
        RMS_values [i] =  librosa.feature.rms(y=np.array(my_data)).mean()
    
    RMS_values_new = {}
    for key, value in RMS_values.items():
        RMS_values_new[key] = np.array(value, dtype= np.float32)
    return RMS_values_new


def apply_music_gain(noise, audio):
    noise_rms = find_RMS_noise(noise)
    noise_rms_arr = np.hstack(list(noise_rms.values()))
    data_rms = find_RMS_data(audio)
    data_rms_arr = np.hstack(list(data_rms.values()))

    current_rms_ratio = data_rms_arr/noise_rms_arr

    length_seg = int(len(current_rms_ratio)/len(user_inputs.sought_ratio))
    master_c = 0
    index = 0 
    combined_ratio_dict = {}
    for i in range(0,len(current_rms_ratio), length_seg):
        factor=user_inputs.sought_ratio[master_c]
        chunk = current_rms_ratio[i:i+length_seg]
        modified_chunk = [x *factor for x in chunk]
        for value in modified_chunk:
            combined_ratio_dict[index]=value
            index+=1
        master_c+=1

    modified_music = {}
    for key, value in audio.items():
        modified_music[key] = combined_ratio_dict[key] * audio[key]

    return modified_music









Data_C = applying_gain_noise.Data_C
Data_I = apply_music_gain(permutation_noise_music.Data_H, permutation_noise_music.Data_G)
