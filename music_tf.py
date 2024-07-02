from imports import *
import user_inputs
import resampling

def transfer_fun(Data):
    if user_inputs.use_filter == False:
        return Data
    else:
        audio_transformed = {}
        system = signal.TransferFunction(user_inputs.filter_num_coeff, user_inputs.filter_dem_coeff).to_discrete(1/user_inputs.downsampling_new_sr)
        for i in range (0, len(Data)):
            _,audio_transformed_single_rec = signal.dlsim(system,Data[i])
            audio_transformed[i] = audio_transformed_single_rec
        return audio_transformed

Data_F = transfer_fun(resampling.Data_E)