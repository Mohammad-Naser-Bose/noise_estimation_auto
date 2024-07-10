from imports import *

recordings_dir = r"audio_files_short"
noise_dir = r"noise_files"
window_size_sec = 4  # in [s]
sampling_freq = 44100  # in [Hz]  
num_epochs=100
train_ratio = .7
val_ratio = .15
downsampling_new_sr = 690 #Ratio=64,128 = 690,344
batch_size = 32
use_tf=True
num_tfs = 3 
tf_types = ["bandpass","lowpass","highpass"]
normalization_flag = True
one_noise_window_flag = True
shortest_music_flag = False
noise_gains = [-3,-5, -7] # dB
sought_ratio = [np.array([4,6,15,1,1]),
                np.array([2,4,12,1,1]),
                np.array([4,7,13,1,1])]#,
                #np.array([3,6,12,1,1])]  # Linear (each line is a different gain profile and one profile has different values for each music file) 

ML_type = "CNN_LSTM"
norm_feature =True

window_len_sample = window_size_sec * sampling_freq
window_len_sample_downsampled = window_size_sec * downsampling_new_sr
noise_files = os.listdir(noise_dir); num_noise_combinations=sum(os.path.isfile(os.path.join(noise_dir,f )) for f in noise_files)
num_noise_files = len(noise_files)
music_files = os.listdir(recordings_dir)
num_music_files = len(music_files)
num_datapoints_nw=len(noise_gains)*len(sought_ratio)*num_noise_files*num_music_files

