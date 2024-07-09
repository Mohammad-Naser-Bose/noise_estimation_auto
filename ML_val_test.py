from imports import *
import splitting_normalization
import ML_train

def ML_validating(model, val_inputs,val_labels):
    reg_criterion = nn.MSELoss()

    val_loss_values = []
    model.eval()
    running_val_loss = 0
    num_val_batches = len(val_inputs)
    errors_val=[]
    all_pred=[]
    all_gt=[]
    for i in range(0, len(val_inputs)):
        my_input = val_inputs[i].requires_grad_(True)
        ground_truth_value = torch.tensor(val_labels[i])        

        predicted_value = model(my_input)
        loss_value = reg_criterion(predicted_value,ground_truth_value)

        running_val_loss += loss_value.item()

        avg_val_loss = running_val_loss / num_val_batches
        val_loss_values.append(avg_val_loss)    

        errors_val.append(((abs(ground_truth_value-predicted_value))/(ground_truth_value))*100)
        all_pred.append(predicted_value)
        all_gt.append(ground_truth_value)
    plotting_results(errors_val,all_pred,all_gt,"Validation")

    return model
def ML_testing(model, test_inputs, test_labels):
    model.eval()
    test_loss = 0
    test_errors=[]
    num_test_batches = len(test_inputs)
    reg_criterion = nn.MSELoss()
    errors_test = []
    all_gt=[]
    all_pred=[]
    for i in range(0, len(test_inputs)):
        my_input = test_inputs[i].requires_grad_(True)
        ground_truth_value = torch.tensor(test_labels[i])        

        predicted_value = model(my_input)
        loss_value = reg_criterion(predicted_value,ground_truth_value)

        test_loss += loss_value.item()
        error = ((abs(ground_truth_value-predicted_value))/(ground_truth_value))*100
        test_errors.append(error)

        #print(f"Orig:{ground_truth_value}, Predicted: {predicted_value}")

        errors_test.append(((abs(ground_truth_value-predicted_value))/(ground_truth_value))*100)
        all_pred.append(predicted_value)
        all_gt.append(ground_truth_value)

    avg_test_loss = test_loss / num_test_batches
    #print(f"Test loss: {avg_test_loss}")

    printing_label = "Testing"
    plotting_results(errors_test,all_pred,all_gt,printing_label)
def plotting_results(error,predictions,gt,printing_label):
    error_ready = [element for array in error for element in array.tolist()]
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

    real_ready = [value.item() for value in gt]
    pred_ready = [value.item() for value in predictions]
    plt.figure(figsize=(10,5))
    plt.plot(real_ready,label="orig")
    plt.plot(pred_ready,label="pred")
    plt.legend()
    plt.xlabel("datapoint")
    plt.ylabel("Noise RMS")
    #plt.title(title)
    #plt.show()
    plt.savefig(f"{printing_label} raw performance.png")

### import model
ML_validating(ML_train.model, splitting_normalization.data_val_xy, splitting_normalization.val_z_norm_l)
ML_testing(ML_train.model, splitting_normalization.data_test_xy, splitting_normalization.test_z_norm_l)