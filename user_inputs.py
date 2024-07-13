from imports import *

start_time = time.time()
recordings_dir = r"audio_files_short"
noise_dir = r"noise_files"
window_size_sec = 4  # in [s]
sampling_freq = 44100  # in [Hz]  
num_epochs=200
train_ratio = .7
val_ratio = .15
downsampling_new_sr = 690 #Ratio=64,128 = 690,344
batch_size = 32
use_tf=False
tf_types = ["lowpass","bandpass","highpass"]
num_tfs = len(tf_types)
normalization_flag = True
one_noise_window_flag = True
shortest_music_flag = True
noise_gains = [i for i in range(0, -20,-1)] # dB
SNRs = [i for i in range(10,1,-1)]             

ML_type = "CNN_LSTM"
norm_feature =True

window_len_sample = window_size_sec * sampling_freq
window_len_sample_downsampled = window_size_sec * downsampling_new_sr
noise_files = os.listdir(noise_dir); num_noise_combinations=sum(os.path.isfile(os.path.join(noise_dir,f )) for f in noise_files)
num_noise_files = len(noise_files)
music_files = os.listdir(recordings_dir)
num_music_files = len(music_files)
num_datapoints_nw=len(noise_gains)*len(SNRs)*num_noise_files*num_music_files

if torch.cuda.is_available():
    print("GPU is active")
else:
    print("noooooooooooo")
