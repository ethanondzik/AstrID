# File: logging.py
# Author: Ethan Ondzik
# Last Changed: Nov 3, 2024

import datetime
import os
import getpass

# Function takes the parameters and formats them into a singular string and returns it
def format_strings(model_history, model_hyperparameters, model_name):
    date = datetime.datetime.now()

    #date format ex: 2024/11/02 12:15:44
    date_string = f'Model run at: {date.strftime("%Y/%m/%d %H:%M:%S")} '
    user_string = f'By user: {getpass.getuser()} ' #should work for windows, linux, and mac
    model_name_string = f'Model: {model_name} '

    #format model history
    history_string = f'Loss and accuracy:\n'
    history_string += f'\tTraining loss: {model_history.history["loss"][-1]}\n' 
    history_string += f'\tValidation loss: {model_history.history["val_loss"][-1]}\n' 
    history_string += f'\tTraining accuracy: {model_history.history["accuracy"][-1]}\n' 
    history_string += f'\tValidation accuracy: {model_history.history["val_accuracy"][-1]}\n\n'

    #format hyperparamters
    hyper_parameter_string = '\nHyper-parameters:\n'
    for i, j in model_hyperparameters.items():
        hyper_parameter_string += f'\t{str(i)}: {str(j)}\n'

    return model_name_string + date_string + user_string + hyper_parameter_string + history_string



#Function writes data about the model being trained and writes this data to a log text file
#The name of the saved model, the time and date of the log being written, the model hyperparameters,
#  and the model history.
def write_to_log(model_history, model_hyperparameters, model_name):
    log_file_name = 'log/model_runs.txt'
    
    if not os.path.exists('log'):
        os.makedirs('log')
    
    log_data = format_strings(model_history, model_hyperparameters, model_name)
    with open(log_file_name, 'a') as log:
        log.write(log_data)


