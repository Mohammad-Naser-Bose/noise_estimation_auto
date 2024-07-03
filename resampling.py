from imports import *
import user_inputs
import loading_data

def resampling(orig_data,label):
    full_resampled_data={}
    for i in range(0, len(orig_data)):
        downsampled_data_temp = librosa.resample(orig_data[i],orig_sr=user_inputs.sampling_freq,target_sr=user_inputs.downsampling_new_sr)
        full_resampled_data[i] = downsampled_data_temp
    return full_resampled_data


Data_B = resampling(loading_data.Data_A,"noise")
Data_E = resampling(loading_data.Data_D,"music")