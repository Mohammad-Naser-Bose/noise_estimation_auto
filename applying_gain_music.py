from imports import *
import user_inputs
import permutation_noise_music

def apply_music_gain(audio_full):
    factors_list = []
    for mod_f in user_inputs.sought_ratio:
        [factors_list.append(val) for n in range(0, user_inputs.num_noise_files*len(user_inputs.noise_gains)) for val in mod_f]
              
    modified_music = {}
    for key, value in audio_full.items():
        modified_music[key] = audio_full[key]/ factors_list[key]
    return modified_music


Data_I = apply_music_gain(permutation_noise_music.Data_G)


# all_noise = applying_gain_noise.Data_C
# all_noise_new={}
# for i,value in all_noise.items():
#     all_noise_new[i]=value[:28000]
# stacked_noise = np.stack(list(all_noise_new.values()),axis=0)
# avg_noise= np.mean(stacked_noise,axis=0)
# fig,axs =plt.subplots(user_inputs.num_music_files,1,figsize=(10,15))
# for i in range(user_inputs.num_music_files):
#     axs[i].plot(Data_I[i], label=f"music#{i+1}");plt.legend()
#     axs[i].plot(avg_noise, label=" avg noise");plt.legend()
    
# plt.tight_layout()
# plt.show()
stop =1
