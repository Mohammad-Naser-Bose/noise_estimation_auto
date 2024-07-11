from imports import *
import user_inputs
import splitting_normalization



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=1, dilation=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.flattened_size= self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size,128)
        self.fc2= nn.Linear(128,32)
        self.fc3= nn.Linear(32,1)

    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.numel()

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=1, dilation=1)
        self.dropout1 = nn.Dropout(0.35)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.flattened_size= self._get_flattened_size()
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=2,batch_first=True)
        self.fc1 = nn.Linear(416,128)
        self.fc2 = nn.Linear(128,32) 
        self.fc3 = nn.Linear(32,1)        
        
    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.numel()

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout1(x)

        x_dim = x.dim()
        if x_dim ==3:
            x=x.permute(0,2,1)
        elif x_dim ==2:
            x=x.permute(1,0)

        x,_ = self.lstm(x)
        #x = self.dropout1(x)
        if x_dim ==3:
            vvv=x.size(0)
            x_reshaped = x.reshape(vvv,-1)
        elif x_dim ==2:
            x_reshaped=x.flatten()
        #x_reshaped = self.dropout1(x_reshaped)
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
def plotting_results(error,predictions,gt,printing_label):
    error_ready= [element for array in error for element in array.tolist()]
    plt.figure(figsize=(10,5))
    capped_data = np.clip(error_ready,a_min=None, a_max=100)
    bins=np.append(np.linspace(0,100,10),np.inf)
    hist, bin_edges = np.histogram(capped_data,bins=bins)
    plt.hist(capped_data,bins=bin_edges,edgecolor="black")
    plt.xlabel("Error [%]")    
    plt.ylabel("Num of datapoints")
    #plt.title(title)
    #plt.show()
    plt.savefig(f"{printing_label} histogram performance.png")

    real_ready = [element for array in gt for element in array.tolist()]
    pred_ready = [element for array in predictions for element in array.tolist()]
    #diff = [a-b for a,b in zip(real_ready,pred_ready)]
    plt.figure(figsize=(10,5))
    plt.plot(real_ready,label="orig")
    plt.plot(pred_ready,label="pred")
    #plt.plot(diff,label="diff")
    plt.legend()
    plt.xlabel("datapoint")
    plt.ylabel("Noise RMS")
    #plt.title(title)
    #plt.show()
    plt.savefig(f"{printing_label} raw performance.png")
def run_ML(train_inputs,train_labels):
    dataset = CustomDataset(train_inputs,train_labels)
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
    reg_criterion = nn.MSELoss()
    model = my_ML_model 
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",factor=0.1, patience=5)

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
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            inputs = inputs.to(torch.float32)
            outputs = model(inputs.squeeze(1))

            loss_value  = reg_criterion(outputs, targets)
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()

            if master_c % 10 ==0:
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
        print("loss is:", loss_value)
        

        avg_train_loss = running_train_loss / num_train_batches
        train_loss_values.append(avg_train_loss)
        scheduler.step(avg_train_loss)

        # ooo = scheduler.get_last_lr()
        # print("Learning Rate:",ooo)

    plotting_performance(train_loss_values,"Training")
    plotting_results(error,predictions,gt, "Training")

    return model, train_loss_values



window_len_sample_downsampled = user_inputs.window_len_sample_downsampled
ML_type = user_inputs.ML_type
batch_size = user_inputs.batch_size
num_epochs = user_inputs.num_epochs

if ML_type == "CNN":
    my_ML_model = CNN()
elif ML_type == "CNN_LSTM":
    my_ML_model = CNN_LSTM()             
device ="cuda"
model,train_loss_values = run_ML(splitting_normalization.data_train_xy,splitting_normalization.train_z_norm_l)

with open("Model.pkl","wb") as file:
    pickle.dump(model,file) 

stop=1