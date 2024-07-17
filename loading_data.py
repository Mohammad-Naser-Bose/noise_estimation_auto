from imports import *
import user_inputs

def loading_data(dir,label):
    if label=="noise":
        num_files = user_inputs.num_noise_files
        cuttoff_samples = user_inputs.cuttoff_noise_sample
    else:
        num_files = user_inputs.num_music_files
        cuttoff_samples = user_inputs.cuttoff_music_sample
    
    files = os.listdir(dir)[:num_files] 

    full_recordings = []
    for i,file in enumerate(files):
        audio_path=os.path.join(dir,file)
        audio_data, sample_rate = librosa.load(audio_path,sr=None)

        audio_data = audio_data[:cuttoff_samples]

        full_recordings.append(audio_data) 
    
    return np.array(full_recordings)


Data_A = loading_data(user_inputs.noise_dir,"noise")
Data_D = loading_data(user_inputs.recordings_dir,"music")
