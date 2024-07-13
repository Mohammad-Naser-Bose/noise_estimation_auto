from imports import *
import user_inputs
import permutation_noise_music
import applying_gain_noise
import resampling

def apply_music_gain(audio_full):
    factors_list = []
    for mod_f in factors:
        if user_inputs.use_tf==True:
            [factors_list.append(val)  for n in range(0, user_inputs.num_noise_files*len(user_inputs.noise_gains)) for nn in range (0, user_inputs.num_tfs) for val in mod_f]
        else:
            [factors_list.append(val)  for n in range(0, user_inputs.num_noise_files*len(user_inputs.noise_gains)) for val in mod_f]

    modified_music = {}
    for key, value in audio_full.items():
        modified_music[key] = audio_full[key]* factors_list[key]
    return modified_music
def find_ratios(music, noise):
    all_noise=[np.array(value) for value in noise.values()]
    noise = np.stack(all_noise,axis=0)
    noise_avg =np.mean(noise,axis=0)

    noise_power = np.mean(noise_avg**2)
    music_power = []
    snrs_as_imported = []
    for i,record in music.items():
        music_power.append(np.mean(record**2))
        snrs_as_imported.append(music_power[i]/noise_power)
    factors_ready = []
    for snr in user_inputs.SNRs:
        fac=[]
        for i,record in enumerate(snrs_as_imported):
            fac.append(np.sqrt(snr/snrs_as_imported[i]))
        fac_arr =np.array(fac)
        factors_ready.append(fac_arr)
    return factors_ready



factors=find_ratios(resampling.Data_E,resampling.Data_B)
Data_I = apply_music_gain(permutation_noise_music.Data_G)


# PLOTTING
### Noise
all_noise = applying_gain_noise.Data_C
all_noise_new={}
for i,value in all_noise.items():
    all_noise_new[i]=value[:user_inputs.window_len_sample_downsampled]
stacked_noise = np.stack(list(all_noise_new.values()),axis=0)
min_noise_avg= stacked_noise[np.where(np.min(np.mean(stacked_noise**2,axis=1)))][0]

### Music
all_music = Data_I
all_music_new={}
min_length =min(len(lst) for lst in all_music.values())
for i,value in all_music.items():
    all_music_new[i]=value[:min_length]
stacked_music = np.stack(list(all_music_new.values()),axis=0)
max_music_avg= stacked_music[np.where(np.max(np.mean(stacked_music**2,axis=1)))][0]

# ### Actual plotting
# repeat_factor = int(np.ceil(len(max_music_avg)/len(min_noise_avg)))
# noise_concat = np.tile(min_noise_avg, repeat_factor)[:len(max_music_avg)]
# x_axis_sample = [i for i in range(0, len(noise_concat))] 
# x_axis_second = [num/user_inputs.downsampling_new_sr for num in x_axis_sample]
# plt.figure(figsize=(10,6))
# plt.plot(x_axis_second, max_music_avg, label=f"Max Music record")
# plt.plot(x_axis_second, noise_concat, label="Min Noise record",alpha=0.5)
# plt.legend(fontsize=14)    
# plt.tight_layout()
# plt.xlabel("Second",fontsize=14)
# plt.ylabel("V",fontsize=14)
# plt.savefig("Worst_SNR.png",bbox_inches="tight",pad_inches=0.1)
# stop =1
