from imports import *

recordings_dir = r"audio_files_short"
noise_dir = r"noise_files"
window_size_sec = 4  # in [s]
sampling_freq = 44100  # in [Hz]  
num_epochs=150
train_ratio = .9
val_ratio = .05
downsampling_new_sr = 690 #Ratio=64,128 = 690,344
batch_size = 1
use_filter=False
filter_num_coeff = [1]
filter_dem_coeff = [1, 1]
normalization_flag = True
noise_gains = [0] # dB
sought_ratio = [5, 10]     # Linear  (noise RMS to music RMS)   # to determine the audio gains 
ML_type = "CNN"
norm_feature =True
num_noise_files = 27
num_music_files = 3


window_len_sample = window_size_sec * sampling_freq
window_len_sample_downsampled = window_size_sec * downsampling_new_sr
noise_files = os.listdir(noise_dir); num_noise_combinations=sum(os.path.isfile(os.path.join(noise_dir,f )) for f in noise_files)


