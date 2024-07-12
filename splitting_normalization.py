from imports import *
import windowing
import user_inputs


def data_splitting(x, y, z):
    keys = list(x.keys())
    random.shuffle(keys)
    train_end = int(user_inputs.train_ratio * len(keys))
    val_end = train_end + int(user_inputs.val_ratio * len(keys))

    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    train_x = {key: x[key] for key in train_keys}
    val_x = {key: x[key] for key in val_keys}
    test_x = {key: x[key] for key in test_keys}

    train_y = {key: y[key] for key in train_keys}
    val_y = {key: y[key] for key in val_keys}
    test_y = {key: y[key] for key in test_keys}

    train_z = {key: z[key] for key in train_keys}
    val_z= {key: z[key] for key in val_keys}
    test_z= {key: z[key] for key in test_keys}

    return train_x, val_x, test_x, train_y, val_y, test_y, train_z, val_z, test_z, train_keys, val_keys, test_keys
def normalization (train_no_norm, val_no_norm, test_no_norm):

    all_windows_training = np.concatenate ([np.array(value) for value in train_no_norm.values()])
    scaler = StandardScaler()
    scaler.fit(all_windows_training.reshape(-1,1))
    normalized_training_windows = {key:scaler.transform(np.array(value).reshape(-1,1)).flatten().tolist() for key, value in train_no_norm.items()}
    normalized_validation_windows = {key:scaler.transform(np.array(value).reshape(-1,1)).flatten().tolist() for key, value in val_no_norm.items()}
    normalized_testing_windows = {key:scaler.transform(np.array(value).reshape(-1,1)).flatten().tolist() for key, value in test_no_norm.items()}

    return normalized_training_windows, normalized_validation_windows, normalized_testing_windows
def FE(data_1, data_2, data_3):
    RMS_values_1 = {}
    for i, recording in enumerate (data_1.items()):
        my_data = recording[1]
        RMS_values_1 [i] = np.sqrt(np.mean((np.array(my_data)**2)))
    temp_arr = np.hstack(list(RMS_values_1.values())).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(temp_arr)
    temp_arr_new = scaler.transform(temp_arr)
    RMS_values_new_1 = {}
    for i in range(0,len(temp_arr_new)):
        RMS_values_new_1[i] = np.array(temp_arr_new[i], dtype= np.float32)

    RMS_values_2 = {}
    for i, recording in enumerate (data_2.items()):
        my_data = recording[1]
        RMS_values_2 [i] = np.sqrt(np.mean((np.array(my_data)**2)))
    temp_arr = np.hstack(list(RMS_values_2.values())).reshape(-1, 1)
    temp_arr_new =scaler.transform(temp_arr)
    RMS_values_new_2 = {}
    for i in range(0,len(temp_arr_new)):
        RMS_values_new_2[i] = np.array(temp_arr_new[i], dtype= np.float32)

    RMS_values_3 = {}
    for i, recording in enumerate (data_3.items()):
        my_data = recording[1]
        RMS_values_3 [i] = np.sqrt(np.mean((np.array(my_data)**2)))
    temp_arr = np.hstack(list(RMS_values_3.values())).reshape(-1, 1)
    temp_arr_new =scaler.transform(temp_arr)
    RMS_values_new_3 = {}
    for i in range(0,len(temp_arr_new)):
        RMS_values_new_3[i] = np.array(temp_arr_new[i], dtype= np.float32)

    return RMS_values_new_1, RMS_values_new_2, RMS_values_new_3, scaler
def FE_perf(data_1, data_2, data_3):
    RMS_values_1 = []
    for i, recording in enumerate (data_1.items()):
        my_data = recording[1]
        RMS_values_1.append(np.sqrt(np.mean((np.array(my_data)**2))))
    
    RMS_values_2 =  []
    for i, recording in enumerate (data_2.items()):
        my_data = recording[1]
        RMS_values_2.append(np.sqrt(np.mean((np.array(my_data)**2))))
    
    RMS_values_3 =  []
    for i, recording in enumerate (data_3.items()):
        my_data = recording[1]
        RMS_values_3.append(np.sqrt(np.mean((np.array(my_data)**2))))

    return RMS_values_1, RMS_values_2, RMS_values_3
def data_prep_for_ML(channel1, channel2):
    keys = sorted(channel1.keys())
    data1_tensors = [torch.tensor(channel1[key]) for key in keys]
    data2_tensors = [torch.tensor(channel2[key]) for key in keys]
    
    data1_batch = torch.stack(data1_tensors)
    data2_batch = torch.stack(data2_tensors)

    data1_batch = data1_batch.unsqueeze(1)
    data2_batch = data2_batch.unsqueeze(1)

    combined_data = torch.cat((data1_batch,data2_batch), dim=1)

    return combined_data

train_x_no_norm, val_x_no_norm, test_x_no_norm, train_y_no_norm, val_y_no_norm, test_y_no_norm, train_o_no_norm, val_o_no_norm, test_o_no_norm, train_keys, val_keys, test_keys = data_splitting (windowing.Data_X, windowing.Data_Y, windowing.Data_O)

train_x_norm, val_x_norm, test_x_norm= normalization (train_x_no_norm, val_x_no_norm, test_x_no_norm)
train_y_norm, val_y_norm, test_y_norm = normalization (train_y_no_norm, val_y_no_norm, test_y_no_norm)
train_o_norm, val_o_norm, test_o_norm = normalization (train_o_no_norm, val_o_no_norm, test_o_no_norm)

train_z_norm, val_z_norm, test_z_norm, scaler  = FE(train_o_norm, val_o_norm, test_o_norm )  

data_train_xy = data_prep_for_ML(train_x_norm, train_y_norm); data_val_xy = data_prep_for_ML(val_x_norm, val_y_norm); data_test_xy = data_prep_for_ML(test_x_norm, test_y_norm)

train_z_norm_l= list(train_z_norm.values()); val_z_norm_l = list(val_z_norm.values()); test_z_norm_l= list(test_z_norm.values())

##### for peerfomance
train_x_FE_norm_l, val_x_FE_norm_l, test_x_FE_norm_l= FE_perf(train_x_norm, val_x_norm, test_x_norm )  
train_y_FE_norm_l, val_y_FE_norm_l, test_y_FE_norm_l= FE_perf(train_y_norm, val_y_norm, test_y_norm )  

stop=1