from imports import *
import user_inputs
import loading_data
import sounddevice as sd

def resampling(orig_data,label):
    full_resampled_data={}
    for i in range(0, len(orig_data)):
        downsampled_data_temp = librosa.resample(orig_data[i],orig_sr=user_inputs.sampling_freq,target_sr=user_inputs.downsampling_new_sr)
        full_resampled_data[i] = downsampled_data_temp
        upsampled = librosa.resample(downsampled_data_temp,orig_sr=690, target_sr=44100)
        # if label == "music":
        #     sd.play(upsampled, samplerate=44100)
        #     sd.wait()
        #     stop=1
    return full_resampled_data


Data_B = resampling(loading_data.Data_A,"noise")
Data_E = resampling(loading_data.Data_D,"music")