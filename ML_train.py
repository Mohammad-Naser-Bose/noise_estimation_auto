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
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1,batch_first=True)
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
def plotting(train_losses, val_losses, train_label, train_pred, val_label, val_pred):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, marker = "o", label = "Training loss")
    plt.plot(val_losses, marker = "o", label = "Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Error_per_epoch.png")    
    
    scaler=MinMaxScaler(feature_range=(1e-5,1))
    labels_norms=scaler.fit_transform(np.array(train_label[-1]).reshape(-1, 1))
    scaler=MinMaxScaler(feature_range=(1e-5,1))
    preds_norms=scaler.fit_transform(np.array(train_pred[-1]).reshape(-1, 1))

    train_label_db = 20*np.log10(labels_norms)
    train_pred_db = 20*np.log10(preds_norms)
    plt.figure(figsize=(10,5))
    plt.plot(train_label_db[:100], marker = "o", label = "True values (training)")
    plt.plot(train_pred_db[:100], marker = "o", label = "Predicted values (training)")
    plt.xlabel("Sample")
    plt.ylabel("Noise RMS [dB]")
    plt.legend()
    plt.savefig("Training1.png")  
    
    err_train = (np.array(train_label[-1])-np.array(train_pred[-1]))/(np.array(train_label[-1]))
    scaler=MinMaxScaler(feature_range=(1e-5,1))
    err_train_norm=scaler.fit_transform(np.array(err_train).reshape(-1, 1))
    err_train_db = 20*np.log10(err_train_norm)
    plt.figure(figsize=(10,5))
    plt.plot(err_train_db[:100], marker = "o", label = "Predicted values (training)")
    plt.axhline(y=np.mean(err_train_db[:100]),color="orange", linestyle="--", label="Avg")
    plt.xlabel("Sample")
    plt.ylabel("Error for Noise RMS [dB]")
    plt.legend()
    plt.savefig("Training2.png") 

    scaler=MinMaxScaler(feature_range=(1e-5,1))
    labels_norms=scaler.fit_transform(np.array(val_label[-1]).reshape(-1, 1))
    scaler=MinMaxScaler(feature_range=(1e-5,1))
    preds_norms=scaler.fit_transform(np.array(val_pred[-1]).reshape(-1, 1))

    val_label_db = 20*np.log10(labels_norms)
    val_pred_db = 20*np.log10(preds_norms)
    plt.figure(figsize=(10,5))
    plt.plot(val_label_db[:100], marker = "o", label = "True values (Validation)")
    plt.plot(val_pred_db[:100], marker = "o", label = "Predicted values (Validation)")
    plt.xlabel("Sample")
    plt.ylabel("Noise RMS [dB]")
    plt.legend()
    plt.savefig("Validation1.png")  
    
    err_val = (np.array(val_label[-1])-np.array(val_pred[-1]))/(np.array(val_label[-1]))
    scaler=MinMaxScaler(feature_range=(1e-5,1))
    err_val_norm=scaler.fit_transform(np.array(err_val).reshape(-1, 1))
    err_val_db = 20*np.log10(err_val_norm)
    plt.figure(figsize=(10,5))
    plt.plot(err_val_db[:100], marker = "o", label = "Predicted values (Validation)")
    plt.axhline(y=np.mean(err_val_db[:100]),color="orange", linestyle="--", label="Avg")
    plt.xlabel("Sample")
    plt.ylabel("Error for Noise RMS [dB]")
    plt.legend()
    plt.savefig("Validation2.png") 


    return()
def save_results(model):
    with open("Model.pkl","wb") as file:
        pickle.dump(model,file)
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
    patience = 10
    best_val_loss = float("inf")
    epochs_no_imrpove = 0
    train_losses = []
    val_losses = []
    train_pred = []
    train_label = []
    val_pred = []
    val_label = []

    for epoch in range(num_epochs):
        epoch_train_preds =[]
        epoch_train_labels =[]

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
            epoch_train_preds.extend(outputs.detach().cpu().numpy().flatten())
            epoch_train_labels.extend(targets.detach().cpu().numpy().flatten())

        avg_train_loss = running_train_loss / len(dataloader_train.dataset)
        train_losses.append(avg_train_loss)
        train_pred.append(epoch_train_preds)
        train_label.append(epoch_train_labels)

        model.eval()
        running_val_loss = 0
        epoch_val_preds =[]
        epoch_val_labels =[]

        with torch.no_grad():
            for val_inputs, val_labels in dataloader_val:
                val_inputs = val_inputs.to(device).to(torch.float32)
                val_labels = val_labels.to(device).to(torch.float32)

                val_outputs = model(val_inputs)
                val_loss_value = reg_criterion(val_outputs, val_labels)

                running_val_loss += val_loss_value.item() * val_inputs.size(0)
                epoch_val_preds.extend(val_outputs.detach().cpu().numpy().flatten())
                epoch_val_labels.extend(val_labels.detach().cpu().numpy().flatten())

        avg_val_loss = running_val_loss / len(dataloader_val.dataset)
        val_losses.append(avg_val_loss)
        val_pred.append(epoch_val_preds)
        val_label.append(epoch_val_labels)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

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

    save_results(model.load_state_dict(best_model_wts))
    plotting(train_losses, val_losses, train_label, train_pred, val_label, val_pred)

    
    return model
    dataset_train = CustomDataset(train_inputs,train_labels)
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size, shuffle=True)

###############
###############
###############
window_len_sample_downsampled = user_inputs.window_len_sample_downsampled
ML_type = user_inputs.ML_type
batch_size = user_inputs.batch_size
num_epochs = user_inputs.num_epochs

if ML_type == "CNN_LSTM":
    my_ML_model = CNN_LSTM()             
device ="cuda"

best_model = run_ML_train_val(splitting_normalization.data_train_xy,splitting_normalization.train_z_norm,splitting_normalization.data_val_xy,splitting_normalization.val_z_norm)
