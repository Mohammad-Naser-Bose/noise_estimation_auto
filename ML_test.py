from imports import *
import splitting_normalization
import ML_train
import user_inputs

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
def plotting_testing(test_preds, test_labels):
    test_label_db = 20*np.log10(np.array(test_labels)/1)
    test_pred_db = 20*np.log10(np.array(test_preds)/1)
    plt.figure(figsize=(10,5))
    plt.plot(test_label_db[:100], marker = "o", label = "True values (Testing)")
    plt.plot(test_pred_db[:100], marker = "o", label = "Predicted values (Testing)")
    plt.xlabel("Sample")
    plt.ylabel("Noise RMS [dB]")
    plt.legend()
    plt.savefig("Testing1.png")  
    
    err_test_db = 20*np.log10((np.array(test_labels) -np.array(test_preds))**2)/(np.array(test_labels)**2)
    err_avg_test_db = np.mean(err_test_db)
    plt.figure(figsize=(10,5))
    plt.plot(err_test_db[:100], marker = "o", label = "Predicted values (validation)")
    plt.axhline(y=err_avg_test_db,color="orange", linestyle="--", label="Avg")
    plt.xlabel("Sample")
    plt.ylabel("Error for Noise RMS [dB]")
    plt.legend()
    plt.savefig("Testing2.png") 
    return 
def run_ML_testing(model, test_inputs, test_labels):
    dataset_test= CustomDataset(test_inputs,test_labels)
    dataloader_test = DataLoader(dataset_test,batch_size=batch_size, shuffle=False)

    model.eval()

    test_preds = []
    test_labels_all = []

    with torch.no_grad():
        for test_inputs, test_labels in dataloader_test:
            test_inputs = test_inputs.to(device).to(torch.float32)
            test_labels = test_labels.to(device).to(torch.float32)

            test_outputs = model(test_inputs)
            test_preds.extend(test_outputs.detach().cpu().numpy().flatten())
            test_labels_all.extend(test_labels.detach().cpu().numpy().flatten())

    plotting_testing(test_preds, test_labels_all)
    return 



###############
###############
###############
window_len_sample_downsampled = user_inputs.window_len_sample_downsampled
ML_type = user_inputs.ML_type
batch_size = user_inputs.batch_size
num_epochs = user_inputs.num_epochs
device = "cuda"
run_ML_testing(ML_train.best_model, splitting_normalization.data_test_xy, splitting_normalization.test_z_norm)
