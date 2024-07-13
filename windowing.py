
from imports import *
import mixing
import reference_preparation
import user_inputs
import pickle



def windowing(signal):
    master_c = 0
    windowed_data = {}
    for i, rec in enumerate (signal.items()):
        my_rec = rec[1]
        num_windows = len(my_rec) // user_inputs.window_len_sample_downsampled
        extra_samples = len(my_rec) % (num_windows*user_inputs.window_len_sample_downsampled)
        truncated_rec = my_rec[extra_samples:]
        rec_windows = np.array_split(truncated_rec, num_windows)
        rec_windows_truncated = [window[:user_inputs.window_len_sample_downsampled] for window in rec_windows]
        for arr in rec_windows_truncated:
            windowed_data[master_c] = arr.tolist()
            master_c+=1
    return windowed_data

# def move_dict_to_gpu(data_dict,device="cuda"):
#     gpu_data_dict = {}
#     for key, value in data_dict.items():
#         tensor_value = torch.tensor(value,dtype=torch.float32).to(device)
#         gpu_data_dict[key]=tensor_value
#     return gpu_data_dict

# Data_Y=windowing(move_dict_to_gpu(reference_preparation.Data_L))
# Data_X=windowing(move_dict_to_gpu(mixing.Data_K))
# Data_O=windowing(move_dict_to_gpu(mixing.Data_J))

Data_Y=windowing(reference_preparation.Data_L)
Data_X=windowing(mixing.Data_K)
Data_O=windowing(mixing.Data_J)