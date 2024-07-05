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
        # self.fc2 = nn.Linear(4096,1024)
        # self.fc3= nn.Linear(1024,256)
        self.fc4= nn.Linear(128,32)
        self.fc5= nn.Linear(32,1)

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
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(13800,10240)
        self.fc2 = nn.Linear(10240,7168)
        self.fc3 = nn.Linear(7168,3072)
        self.fc4 = nn.Linear(3072,1024)
        self.fc5 = nn.Linear(1024,256)
        self.fc6 = nn.Linear(256,64)
        self.fc7 = nn.Linear(64,1)

    def forward(self,x):
        batch_size, dim1, dim2 = x.size()
        x = x.view(batch_size,dim1 * dim2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        
        return x
class CNN_LSTM_b(nn.Module):
    def __init__(self):
        super(CNN_LSTM_b, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=3, padding=1, dilation=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=1, dilation=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.flattened_size= self._get_flattened_size()
        self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=512, num_layers=1)
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,64)        
        self.fc3 = nn.Linear(64,1)
        
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

        x,_ = self.lstm(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)
        return x
class CNN_LSTM_a(nn.Module):
    def __init__(self):
        super(CNN_LSTM_a, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=5, padding=1, dilation=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU(negative_slope=0.01)
        self.flattened_size= self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size,4096)
        self.lstm = nn.LSTM(input_size=4096, hidden_size=4096, num_layers=2)
        self.fc2= nn.Linear(4096,1024)
        self.fc3= nn.Linear(1024,128)
        self.fc4= nn.Linear(128,1)
        
    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x = self.pool(self.relu(self.conv1(x)))
        return x.numel()

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)

        x = self.relu(self.fc1(x))
        x,_ = self.lstm(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
class CNN_LSTM_c(nn.Module):
    def __init__(self):
        super(CNN_LSTM_c, self).__init__()
        self.lstm = nn.LSTM(input_size=window_len_sample_downsampled, hidden_size=500, num_layers=5)

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=2, stride=2, padding=1, dilation=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU(negative_slope=0.01)
        self.flattened_size= self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3= nn.Linear(64,1)
        
    def _get_flattened_size(self):
        x = torch.zeros(1,2,window_len_sample_downsampled) # one sample regardless the batch size, num channels, num timepoints
        x,_  = self.lstm(x)
        x = self.pool(self.relu(self.conv1(x)))
        return x.numel()

    def forward(self,x):
        x,_  = self.lstm(x)
        x = self.pool(self.relu(self.conv1(x)))

        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.relu = nn.ReLU(negative_slope=0.01)
        self.fc1 = nn.Linear(74304,4096)
        self.fc2 = nn.Linear(4096,2048)
        self.fc3 = nn.Linear(2048,1024)
        self.fc4 = nn.Linear(1024,512)
        self.fc5= nn.Linear(512,256)
        self.fc6= nn.Linear(256,128)
        self.fc7= nn.Linear(128,64)
        self.fc8= nn.Linear(64,32)
        self.fc9= nn.Linear(32,16)
        self.fc10= nn.Linear(16,8)
        self.fc11= nn.Linear(8,4)
        self.fc12= nn.Linear(4,2)
        self.fc13= nn.Linear(2, 1)
        
    def forward(self,x):
        x_dim = x.dim()
        if x_dim==3:
            x=x.view(x.size(0), -1)
        else:
            x=x.view(-1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.relu(self.fc10(x))
        x = self.relu(self.fc11(x))
        x = self.relu(self.fc12(x))
        x = self.fc13(x)
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
def plotting_results_general_training(error,predictions,gt,printing_label):
    error_ready = [element for array in error for element in array.tolist()]
    # plt.figure(figsize=(10,5))
    # plt.hist(error_ready,bins=100)
    # plt.xlabel("Error [%]")
    # plt.ylabel("Num of datapoints")
    # #plt.title(title)
    # #plt.show()
    # plt.savefig(f"{printing_label} histogram performance.png")

    real_ready = [element for array in gt for element in array.tolist()]
    pred_ready = [element for array in predictions for element in array.tolist()]
    diff = [a-b for a,b in zip(real_ready,pred_ready)]
    plt.figure(figsize=(10,5))
    plt.plot(real_ready,label="orig")
    plt.plot(pred_ready,label="pred")
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
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",factor=0.1, patience=10)

    train_loss_values = []
    error=[]
    predictions=[]
    gt=[]
    for epoch in range(num_epochs):
        print("-----------------------------")
        print("epoch:", epoch)
        model.train()
        running_train_loss = 0
        num_train_batches = len(train_inputs)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(torch.float32)
            outputs = model(inputs.squeeze(1))

            loss_value  = reg_criterion(outputs, targets)
            loss_value.backward()
            optimizer.step()
            running_train_loss += loss_value.item()

            print("real:", targets.detach().cpu().numpy().flatten())
            print("pred:", outputs.detach().cpu().numpy().flatten())
            print("-------")
            

            

            if epoch == num_epochs-1:
                ground_truth_values = targets.detach().cpu().numpy().flatten()
                predicted_values = outputs.detach().cpu().numpy().flatten()
                error.append(((abs(ground_truth_values-predicted_values))/(ground_truth_values))*100)

                predictions.append(predicted_values)
                gt.append(ground_truth_values)
        print("loss is:", loss_value)
        

        avg_train_loss = running_train_loss / num_train_batches
        train_loss_values.append(avg_train_loss)
        #scheduler.step(avg_train_loss)

        #print(scheduler.get_last_lr())

    plotting_performance(train_loss_values,"Training")

    return model




window_len_sample_downsampled = user_inputs.window_len_sample_downsampled
ML_type = user_inputs.ML_type
batch_size = user_inputs.batch_size
num_epochs = user_inputs.num_epochs
if ML_type == "CNN":
    my_ML_model = CNN()
elif ML_type == "NN":
    my_ML_model = NN()
elif ML_type == "CNN_LSTM_a":
    my_ML_model = CNN_LSTM_a()      
elif ML_type == "CNN_LSTM_b":
    my_ML_model = CNN_LSTM_b()            
elif ML_type == "CNN_LSTM_c":
    my_ML_model = CNN_LSTM_c()    
elif ML_type == "FC":
    my_ML_model = FC()     

model = run_ML(splitting_normalization.data_train_xy,splitting_normalization.train_z_norm_l)

    #### things different from previous: the way the gain is applied, the noise feature is different, and only one window of each noise file was considered.
    ####### TRY SAME STANDARIZATION (MIN-MAX), AND [-1 1] NOT [0 1]
