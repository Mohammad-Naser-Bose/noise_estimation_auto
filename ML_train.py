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
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(64)

        self.dropout1 = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.flattened_size= self._get_flattened_size()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(832,256)
        self.fc2 = nn.Linear(256,64) 
        self.fc3 = nn.Linear(64,1)        
        
    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  
        return x.numel()

    def forward(self,x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))    
        #x = self.dropout1(x)

        x_dim = x.dim()
        if x_dim ==3:
            x=x.permute(0,2,1)
        elif x_dim ==2:
            x=x.permute(1,0)

        x,_ = self.lstm(x)

        if x_dim ==3:
            vvv=x.size(0)
            x_reshaped = x.reshape(vvv,-1)
        elif x_dim ==2:
            x_reshaped=x.flatten()

        x = self.relu(self.bn_fc1(self.fc1(x_reshaped)))
        x = self.relu(self.bn_fc2(self.fc2(x)))
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

    # real_ready_unscaled = splitting_normalization.scaler.inverse_transform(np.array(real_ready).reshape((len(real_ready),1)))
    # pred_ready_unscaled = splitting_normalization.scaler.inverse_transform(np.array(pred_ready).reshape((len(pred_ready),1)))

    # real_ready_g = [element for array in real_ready_unscaled for element in array.tolist()]
    # pred_ready_g = [element for array in pred_ready_unscaled for element in array.tolist()]


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
def run_ML(train_inputs,train_labels):
    dataset = CustomDataset(train_inputs,train_labels)
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
    reg_criterion = nn.MSELoss()
    model = my_ML_model 
    #model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=0.001)#,weight_decay=0.000001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",factor=0.1, patience=5)

    train_loss_values = []
    error=[]
    predictions=[]
    gt=[]
    master_c = 0
    for epoch in range(num_epochs):
        print("-----------------------------")
        print("epoch:", epoch)
        model.train()
        running_train_loss = 0
        num_train_batches = len(train_inputs)
        

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            #inputs = inputs.to(device)
            #targets = targets.to(device)
            optimizer.zero_grad()
            inputs = inputs.to(torch.float32)
            outputs = model(inputs.squeeze(1))

            loss_value  = reg_criterion(outputs, targets)
            weighted_loss=(loss_value*weights).mean()
            weighted_loss.backward()
            optimizer.step()
            running_train_loss += weighted_loss.item()

            if master_c % 30 ==0:
                print("real:", targets.detach().cpu().numpy().flatten())
                print("pred:", outputs.detach().cpu().numpy().flatten())
                print("-------")
            master_c +=1
            
            if epoch == num_epochs-1:
                ground_truth_values = targets.detach().cpu().numpy().flatten()
                predicted_values = outputs.detach().cpu().numpy().flatten()
                error.append(((abs(ground_truth_values-predicted_values))/(ground_truth_values))*100)

                predictions.append(predicted_values)
                gt.append(ground_truth_values)
        print("loss is:", weighted_loss)
        

        avg_train_loss = running_train_loss / num_train_batches
        train_loss_values.append(avg_train_loss)
        #scheduler.step(avg_train_loss)

        # ooo = scheduler.get_last_lr()
        # print("Learning Rate:",ooo)

    plotting_performance(train_loss_values,"Training")
    #plotting_results(error,predictions,gt, "Training")
    pm_training = plotting_db(error,predictions,gt, "Training")
    save_results(model, train_loss_values, error, predictions,gt)
    return model, train_loss_values, pm_training
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


window_len_sample_downsampled = user_inputs.window_len_sample_downsampled
ML_type = user_inputs.ML_type
batch_size = user_inputs.batch_size
num_epochs = user_inputs.num_epochs

if ML_type == "CNN_LSTM":
    my_ML_model = CNN_LSTM()             
#device ="cuda"
weights = torch.ones(len(splitting_normalization.train_z_norm_l))#.to(device)
for i in range(len(splitting_normalization.train_z_norm_l)):
    weights[i]*=1
    # if splitting_normalization.train_z_norm_l[i][0]>.33: #or splitting_normalization.train_z_norm_l[i][0]<.05:
    #     weights[i]*=2
model,train_loss_values, pm_training = run_ML(splitting_normalization.data_train_xy,splitting_normalization.train_z_norm_l)
