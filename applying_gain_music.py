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
    RMS_values =RMS_values
    return RMS_values


def apply_music_gain(noise, audio_full, audio_short):
    noise_rms_avg = find_RMS_noise(noise)
    data_rms = find_RMS_data(audio_short)

    # current_rms_ratio = noise_rms_avg/data_rms
    # modified_factors = []
    # for val in user_inputs.sought_ratio:
    #     modified_factors.append(current_rms_ratio * val)
    modified_factors = []
    modified_factors = [np.array([4,6,15]),np.array([4,6,15])]


    modified_factors_list = []
    for mod_f in modified_factors:
        [modified_factors_list.append(val) for n in range(0, user_inputs.num_noise_files*len(user_inputs.noise_gains)) for val in mod_f]
              
    
    # gain based on ratio
    modified_music = {}
    for key, value in audio_full.items():
        modified_music[key] = audio_full[key]/ modified_factors_list[key]
    return modified_music


Data_I = apply_music_gain(permutation_noise_music.Data_H, permutation_noise_music.Data_G,music_tf.Data_F)


# all_noise = applying_gain_noise.Data_C
# all_noise_new={}
# for i,value in all_noise.items():
#     all_noise_new[i]=value[:28000]
# stacked_noise = np.stack(list(all_noise_new.values()),axis=0)
# avg_noise= np.mean(stacked_noise,axis=0)
# fig,axs =plt.subplots(3,1,figsize=(10,15))
# for i in range(3):
#     axs[i].plot(Data_I[i], label=f"music#{i+1}");plt.legend()
#     axs[i].plot(avg_noise, label=" avg noise");plt.legend()
    
# plt.tight_layout()
# plt.show()
# stop =1
