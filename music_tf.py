from imports import *
import user_inputs
import resampling
from scipy.signal import welch
from scipy.signal import butter, lfilter, freqz



def plotting_tf(signal_before, signal_after):
    freq_before,psd_before = welch(signal_before, 690, nperseg=128)
    freq_after,psd_after = welch(signal_after, 690, nperseg=128)
    plt.figure(figsize=(10,30))
    plt.semilogy(freq_before,psd_before,label="before")
    plt.semilogy(freq_after,psd_after,label="after",alpha=.5)
    plt.legend()
    plt.show()
    stop=1
    return
def butter_filter(data, cutoff, fs,order, type):
    if type == "bandpass":
        nyquist = 0.5 * fs
        cuts=[]
        for cut in cutoff:
            normal_cuttoff = cut / nyquist
            cuts.append(normal_cuttoff)
        b,a = butter(order,cuts, btype=type,analog=False)
        y = lfilter(b,a,data)
    else:
        nyquist = 0.5 * fs
        normal_cuttoff = cutoff / nyquist
        b,a = butter(order,normal_cuttoff, btype=type,analog=False)
        y = lfilter(b,a,data)
    return y
def transfer_fun(Data):
    if user_inputs.use_tf == False:
        return Data
    else:
        audio_transformed = {}
        master_c = 0
        for i, type in enumerate(user_inputs.tf_types):
            if type=="bandpass":
                freq = [172, 225]
            else:
                freq = 172
            for i in range (0, len(Data)):
                audio_transformed_single_rec = butter_filter(Data[i], freq, 690,5,type)
                audio_transformed[master_c] = np.squeeze(audio_transformed_single_rec)
                plotting_tf(Data[i],audio_transformed[master_c]) 
                master_c+=1
        return audio_transformed

Data_F = transfer_fun(resampling.Data_E)

STOP=1