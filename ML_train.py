from imports import *
import user_inputs
import splitting_normalization

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=1, dilation=1)    
        self.bn1 = nn.BatchNorm1d(8)

        self.dropout1 = nn.Dropout(0.3)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.flattened_size= self._get_flattened_size()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=3,batch_first=True)
        self.fc1 = nn.Linear(3264,1024)
        self.fc2 = nn.Linear(1024,64) 
        self.fc3 = nn.Linear(64,1)        
        
    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(self.relu(self.conv3(x)))
        return x.numel()

    def forward(self,x):
        x = self.relu(self.conv1(x))#; x = self.bn1()
        x = self.relu(self.conv2(x))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout1(x)
        
        x=x.permute(0,2,1)
        x,_ = self.lstm(x)

        x_reshaped = x.reshape(x.size(0),-1)
        x = self.relu(self.fc1(x_reshaped))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class CustomDataset(Dataset):
    def __init__(self,inputs,labels):
        self.inputs = inputs
        self.labels = labels
    def __len__(self):
        return(len(self.inputs))
    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label = self.labels[idx]
        return input_data, label
def plotting_performance(loss_values,title):
    plt.figure(figsize=(10,5))
    plt.plot(range(1,num_epochs+1), loss_values, marker = "o", label = "Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    #plt.show()
    plt.savefig("Training error_per_epoch.png")
def plotting_db(error,predictions,gt,printing_label):
    real_ready = [element for array in gt for element in array.tolist()]
    pred_ready = [element for array in predictions for element in array.tolist()]
    error_ready= [element for array in error for element in array.tolist()]

    real_noise_db = 20*np.log10(np.array(real_ready)/1)
    pred_noise_db= 20*np.log10(np.array(pred_ready)/1)   
    diff_1 = real_noise_db -pred_noise_db

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10))
    ax1.plot(real_noise_db[:100],label="Original")
    ax1.plot(pred_noise_db[:100],label="Prediction")
    ax1.legend()
    ax1.set_xlabel("Datapoint")
    ax1.set_ylabel("Noise RMS (dB)")
    ax2.grid(True)
    #ax1.show() 

    ax2.plot(np.abs(diff_1[:100]),label="Difference between actual and predicted noise",color="black")
    mean = np.mean(np.abs(diff_1[:100]))
    ax2.axhline(y=mean,color="orange", linestyle="--", label="Avg")
    ax2.legend()
    ax2.set_xlabel("Datapoint")
    ax2.set_ylabel("|Noise RMS (dB)|")
    ax2.grid(True)
    #ax1.show()
    plt.tight_layout()
    plt.savefig(f"{printing_label} raw performance.png")    

    ##############
    # SNR_real_db = 20*np.log10(np.array(splitting_normalization.train_y_FE_norm_l)/1)
    # SNR_pred_db = 20*np.log10(np.array(splitting_normalization.train_y_FE_norm_l)/1)
    # diff = SNR_real_db - SNR_pred_db

    # fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10))
    # ax1.plot(SNR_real_db[:100],label="Original")
    # ax1.plot(SNR_pred_db[:100],label="Prediction")
    # ax1.legend()
    # ax1.set_xlabel("Datapoint")
    # ax1.set_ylabel("SNR (dB)")
    # ax1.grid(True)
    # #ax1.show() 

    # ax2.plot(diff[:1000],label="Difference between actual and predicted noise",color="black")
    # ax2.legend()
    # ax2.set_xlabel("Datapoint")
    # ax2.set_ylabel("SNR (dB)")
    # #ax1.show()
    # plt.tight_layout()
    # ax2.grid(True)
    # plt.savefig(f"{printing_label} raw performance2.png")  


    
    return(np.mean(diff_1))
def run_ML_train(train_inputs,train_labels):#
    dataset = CustomDataset(train_inputs,train_labels)
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
    reg_criterion = nn.MSELoss(reduction='mean')
    model = my_ML_model 
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=.001)

    train_loss_values = []
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0


        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device).to(torch.float32)
            targets = targets.to(device).to(torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value  = reg_criterion(outputs, targets)
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()

    return model, train_loss_values
def save_results(model, train_loss_values, error, predictions,gt):
    with open("train_loss_values.pkl","wb") as file:
        pickle.dump(train_loss_values,file)
    with open("error.pkl","wb") as file:
        pickle.dump(error,file)
    with open("predictions.pkl","wb") as file:
        pickle.dump(predictions,file)
    with open("gt.pkl","wb") as file:
        pickle.dump(gt,file)
    return
def run_ML_train_val(train_inputs,train_labels, val_inputs, val_labels):#
    dataset_train = CustomDataset(train_inputs,train_labels)
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size, shuffle=True)
    dataset_val = CustomDataset(train_inputs,train_labels)
    dataloader_val = DataLoader(dataset_val,batch_size=batch_size, shuffle=True)

    reg_criterion = nn.MSELoss(reduction='mean')
    model = my_ML_model 
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001, weight_decay=0.000001)
    patience = 5
    best_val_loss = float("inf")
    epochs_no_imrpove = 0

    train_loss_values = []
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0


        for batch_idx, (inputs, targets) in enumerate(dataloader_train):
            inputs = inputs.to(device).to(torch.float32)
            targets = targets.to(device).to(torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value  = reg_criterion(outputs, targets)
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()
        avg_train_loss = running_train_loss / len(dataloader_train.dataset)

        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for val_inputs, val_labels in dataloader_val:
                val_inputs = val_inputs.to(device).to(torch.float32)
                val_labels = val_labels.to(device).to(torch.float32)

                val_outputs = model(val_inputs)
                val_loss_value = reg_criterion(val_outputs, val_labels)

                running_val_loss += val_loss_value.item() * val_inputs.size(0)

        avg_val_loss = running_val_loss / len(dataloader_val.dataset)
        print(f"Epoch{epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss =avg_val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve +=1
        
        if epochs_no_improve == patience:
            print("Early stopping triggered")
            break
    return model.load_state_dict(best_model_wts)



window_len_sample_downsampled = user_inputs.window_len_sample_downsampled
ML_type = user_inputs.ML_type
batch_size = user_inputs.batch_size
num_epochs = user_inputs.num_epochs

if ML_type == "CNN_LSTM":
    my_ML_model = CNN_LSTM()             
device ="cuda"

model,train_loss_values= run_ML_train(splitting_normalization.data_train_xy,splitting_normalization.train_z_norm)
#best_model = run_ML_train_val(splitting_normalization.data_train_xy,splitting_normalization.train_z_norm,splitting_normalization.data_val_xy,splitting_normalization.val_z_norm)
