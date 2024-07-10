from imports import *
import user_inputs
import resampling
from scipy.signal import welch

def plotting_tf(signal_before, signal_after):
    freq_before,psd_before = welch(signal_before, 44100, nperseg=1024)
    freq_after,psd_after = welch(signal_after, 44100, nperseg=1024)
    plt.figure(figsize=(10,30))
    plt.semilogy(freq_before,psd_before,label="before")
    plt.semilogy(freq_after,psd_after,label="after",alpha=.5)
    plt.legend()
    plt.show()
    stop=1
    return
def transfer_fun(Data):
    if user_inputs.use_tf == False:
        return Data
    else:
        audio_transformed = {}
        master_c = 0
        for filter_i in range(0,user_inputs.num_tfs):
            system = signal.TransferFunction(user_inputs.num_coeffs[filter_i], user_inputs.den_coeffs[filter_i]).to_discrete(1/user_inputs.downsampling_new_sr)
            for i in range (0, len(Data)):
                _,audio_transformed_single_rec = signal.dlsim(system,Data[i])
                audio_transformed[master_c] = audio_transformed_single_rec
                #plotting_tf(Data[i],audio_transformed_single_rec[:,0]) # CHECK  
                master_c+=1
        return audio_transformed



Data_F = transfer_fun(resampling.Data_E)


STOP=1