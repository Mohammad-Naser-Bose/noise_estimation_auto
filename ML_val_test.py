from imports import *
import splitting_normalization
import ML_train
import user_inputs


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

        my_input = my_input.to(device)
        ground_truth_value = ground_truth_value.to(device) 

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
    pm_test = plotting_results_test_db(errors_test,all_pred,all_gt,printing_label)
    return pm_test

def plotting_results_test_db(error,predictions,gt,printing_label):
    real_ready = [element for array in gt for element in array.tolist()]
    pred_ready = [element for array in predictions for element in array.tolist()]
    error_ready= [element for array in error for element in array.tolist()]

    real_ready_unscaled = splitting_normalization.scaler.inverse_transform(np.array(real_ready).reshape((len(real_ready),1)))
    pred_ready_unscaled = splitting_normalization.scaler.inverse_transform(np.array(pred_ready).reshape((len(pred_ready),1)))

    real_ready_g = [element for array in real_ready_unscaled for element in array.tolist()]
    pred_ready_g = [element for array in pred_ready_unscaled for element in array.tolist()]


    real_noise_db = 20*np.log10(np.array(real_ready_g)/1)
    pred_noise_db= 20*np.log10(np.array(pred_ready_g)/1)   
    diff_1 = real_noise_db -pred_noise_db

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10))
    ax1.plot(real_noise_db[:100],label="Original")
    ax1.plot(pred_noise_db[:100],label="Prediction")
    ax1.legend()
    ax1.set_xlabel("Datapoint")
    ax1.set_ylabel("Noise RMS (dB)")
    #ax1.show() 

    ax2.plot(diff_1[:100],label="Difference between actual and predicted noise",color="black")
    ax2.legend()
    ax2.set_xlabel("Datapoint")
    ax2.set_ylabel("Noise RMS (dB)")
    #ax1.show()
    plt.tight_layout()
    plt.savefig(f"{printing_label} raw performance.png")    
    ##############
    SNR_real_db = 20*np.log10(np.array(splitting_normalization.test_y_FE_norm_l)/1)
    SNR_pred_db = 20*np.log10(np.array(splitting_normalization.test_y_FE_norm_l)/1)
    diff = SNR_real_db - SNR_pred_db

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10))
    ax1.plot(SNR_real_db,label="Original")
    ax1.plot(SNR_pred_db,label="Prediction")
    ax1.legend()
    ax1.set_xlabel("Datapoint")
    ax1.set_ylabel("SNR (dB)")
    #ax1.show() 

    ax2.plot(diff,label="Difference between actual and predicted noise",color="black")
    ax2.legend()
    ax2.set_xlabel("Datapoint")
    ax2.set_ylabel("SNR (dB)")
    #ax1.show()
    plt.tight_layout()
    plt.savefig(f"{printing_label} raw performance2.png")  
    return np.mean(diff_1)
def generate_report(time, num_music, num_noise, noise_gains, SNRs, tf_types,model_type, training_loss,pm_train,pm_val,pm_test):
    with open ("Results report","w") as file:
        file.write(f"Num. of Music Records: {num_music}\n")
        file.write(f"Num. of Noise Clips: {num_noise}\n")
        file.write(f"noise_gains: {noise_gains}\n")
        file.write(f"SNRs: {SNRs}\n")
        file.write(f"tf_types: {tf_types}\n")
        file.write(f"model type: {model_type}\n")
        file.write(f"training loss: {training_loss}\n")
        file.write(f"pm-train: {pm_train}\n")
        file.write(f"pm-val: {pm_val}\n")     
        file.write(f"pm-test: {pm_test}\n")    
    return 

### import model
device = "cuda"
pm_test=ML_testing(ML_train.model, splitting_normalization.data_test_xy, splitting_normalization.test_z_norm_l)

end_time = time.time()
full_time = end_time-user_inputs.start_time
generate_report(full_time,user_inputs.num_music_files,user_inputs.num_noise_files, user_inputs.noise_gains, user_inputs.SNRs, user_inputs.tf_types, user_inputs.ML_type,ML_train.train_loss_values[:-1],ML_train.pm_training,pm_val,pm_test)