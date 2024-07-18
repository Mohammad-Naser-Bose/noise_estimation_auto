from imports import *

recordings_dir = r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\audio_files_short"
noise_dir = r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\noise_files"
window_size_sec = 4  # in [s]
sampling_freq = 44100  # in [Hz]  
num_epochs=25
train_ratio = .7
val_ratio = .15
downsampling_new_sr = 690 #Ratio=64,128 = 690,344
batch_size = 1

use_tf=True
if use_tf == True:
    tf_types = ["lowpass","bandpass","highpass"]
else:
    tf_types = ["None"]
num_tfs = len(tf_types)

noise_gains = [i for i in np.arange(-9, 12, 3)] # dB   
SNRs = [2, 1.75, 1.5, 1.25, 1, .75, .5, .25]# Linear       
cuttoff_noise_sample = 1*sampling_freq
cuttoff_music_sample = 12*sampling_freq
ML_type = "CNN_LSTM"

window_len_sample = window_size_sec * sampling_freq
window_len_sample_downsampled = window_size_sec * downsampling_new_sr
noise_files = os.listdir(noise_dir); num_noise_combinations=sum(os.path.isfile(os.path.join(noise_dir,f )) for f in noise_files)
num_noise_files = len(noise_files)
music_files = os.listdir(recordings_dir)
num_music_files = len(music_files)
num_datapoints_nw=len(noise_gains)*len(SNRs)*num_noise_files*num_music_files

