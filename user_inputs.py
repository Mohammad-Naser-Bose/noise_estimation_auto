recordings_dir = r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\audio_files_short"
noise_dir = r"C:\Users\mn1059928\OneDrive - Bose Corporation\Desktop\noise_files"
window_size_sec = 4  # in [s]
sampling_freq = 44100  # in [Hz]  
num_epochs=150
train_ratio = .9
val_ratio = .05
downsampling_new_sr = 690   #Ratio=64,128 = 690,344
batch_size = 1
use_filter=False
filter_num_coeff = [1]
filter_dem_coeff = [1, 1]
normalization_flag = True
noise_gains = [0] # dB
ML_type = "CNN"
norm_feature =True
sought_ratio = [2, 5, 10]         # data to noise          #audio_gains = [-10,-50,-90] #[i for i in np.arange(-100,-210,-10)]  