from imports import *
import user_inputs
import permutation_noise_music
import applying_gain_noise

def apply_music_gain(audio_full):
    factors_list = []
    for mod_f in user_inputs.sought_ratio:
        [factors_list.append(val)  for n in range(0, user_inputs.num_noise_files*len(user_inputs.noise_gains)) for nn in range (0, user_inputs.num_tfs) for val in mod_f]
              
    modified_music = {}
    for key, value in audio_full.items():
        modified_music[key] = audio_full[key]/ factors_list[key]
    return modified_music


Data_I = apply_music_gain(permutation_noise_music.Data_G)


# PLOTTING
### Noise
all_noise = applying_gain_noise.Data_C
all_noise_new={}
for i,value in all_noise.items():
    all_noise_new[i]=value[:user_inputs.window_len_sample_downsampled]
stacked_noise = np.stack(list(all_noise_new.values()),axis=0)
min_noise_avg= stacked_noise[np.where(np.min(np.mean(stacked_noise,axis=1)))][0]

### Music
all_music = Data_I
all_music_new={}
min_length =min(len(lst) for lst in all_music.values())
for i,value in all_music.items():
    all_music_new[i]=value[:min_length]
stacked_music = np.stack(list(all_music_new.values()),axis=0)
max_music_avg= stacked_music[np.where(np.max(np.mean(stacked_music,axis=1)))][0]

### Actual plotting
repeat_factor = int(np.ceil(len(max_music_avg)/len(min_noise_avg)))
noise_concat = np.tile(min_noise_avg, repeat_factor)[:len(max_music_avg)]
x_axis_sample = [i for i in range(0, len(noise_concat))] 
x_axis_second = [num/user_inputs.downsampling_new_sr for num in x_axis_sample]
plt.figure(figsize=(10,6))
plt.plot(x_axis_second, max_music_avg, label=f"Max Music record")
plt.plot(x_axis_second, noise_concat, label="Min Noise record",alpha=0.5)
plt.legend(fontsize=14)    
plt.tight_layout()
plt.xlabel("Second",fontsize=14)
plt.ylabel("V",fontsize=14)
plt.savefig("noise_music_raw.png",bbox_inches="tight",pad_inches=0.1)
stop =1
