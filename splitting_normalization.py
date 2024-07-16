from imports import *
import windowing
import user_inputs


def data_splitting(x, y, z):
    keys = [i for i in range(0,len(x))]
    random.shuffle(keys)
    train_end = int(user_inputs.train_ratio * len(keys))
    val_end = train_end + int(user_inputs.val_ratio * len(keys))

    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    train_x = np.array([x[key] for key in train_keys])
    val_x = np.array([x[key] for key in val_keys])
    test_x = np.array([x[key] for key in test_keys])

    train_y = np.array([y[key] for key in train_keys])
    val_y = np.array([y[key] for key in val_keys])
    test_y = np.array([y[key] for key in test_keys])

    train_z = np.array([z[key] for key in train_keys])
    val_z = np.array([z[key] for key in val_keys])
    test_z = np.array([z[key] for key in test_keys])

    return train_x, val_x, test_x, train_y, val_y, test_y, train_z, val_z, test_z, train_keys, val_keys, test_keys
def normalization (train_no_norm, val_no_norm, test_no_norm):
    scaler = StandardScaler()
    scaler.fit(train_no_norm)
    normalized_training_windows = np.array(scaler.transform(train_no_norm))
    normalized_validation_windows = np.array(scaler.transform(val_no_norm))
    normalized_testing_windows = np.array(scaler.transform(test_no_norm))
    return normalized_training_windows, normalized_validation_windows, normalized_testing_windows
def FE(data_1, data_2, data_3):
    RMS_values_1 = np.zeros(shape=(len(data_1)))
    master_c=0
    for recording in data_1:
        RMS_values_1[master_c]=np.sqrt(np.mean(recording**2))
        master_c+=1

    RMS_values_2 = np.zeros(shape=(len(data_2)))
    master_c=0
    for recording in data_2:
        RMS_values_2[master_c]=np.sqrt(np.mean(recording**2))
        master_c+=1

    RMS_values_3 = np.zeros(shape=(len(data_3)))
    master_c=0
    for recording in data_3:
        RMS_values_3[master_c]=np.sqrt(np.mean(recording**2))
        master_c+=1

    return RMS_values_1, RMS_values_2, RMS_values_3
def data_prep_for_ML(channel1, channel2):
    keys = sorted([i for i in range(0,len(channel1))])
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

train_z_norm, val_z_norm, test_z_norm  = FE(train_o_norm, val_o_norm, test_o_norm )  

data_train_xy = data_prep_for_ML(train_x_norm, train_y_norm)
data_val_xy = data_prep_for_ML(val_x_norm, val_y_norm)
data_test_xy = data_prep_for_ML(test_x_norm, test_y_norm)


stop=1