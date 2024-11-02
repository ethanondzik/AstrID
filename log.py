# File: logging.py
# Author: Ethan Ondzik
# Last Changed: Nov 2, 2024

import datetime
import os
import getpass

def write_to_log(model_history, model_hyperparameters, model_name):
    log_file_name = 'log/model_runs.txt'
    
    date = datetime.datetime.now()

    #date format ex: 2024/11/02 12:15:44
    date_string = 'Model run at: ' + date.strftime('%Y/%m/%d %H:%M:%S')
    user_string = 'By user: ' + getpass.getuser() #should work for windows, linux, and mac

    #format model history
    history_string = f'''Loss and accuracy:
    Training loss: {model_history.history['loss'][-1]}
    Validation loss: {model_history.history['val_loss'][-1]}
    Training accuracy: {model_history.history['accuracy'][-1]}
    Validation accuracy: {model_history.history['val_accuracy'][-1]}\n
    '''

    #format hyperparamters
    hyper_parameter_string = 'Hyper-parameters:\n'
    for i, j in model_hyperparameters.items():
        hyper_parameter_string += '\t' + str(i) + ': ' + str(j) + '\n'

    #write information to log_file_name
    if not os.path.exists('log'):
        os.makedirs('log')
    
    with open(log_file_name, 'a') as log:
        log.write('\n' + 'Model: ' + model_name + ' ' + date_string + ' ' + user_string + '\n')
        log.write(hyper_parameter_string)
        log.write(history_string)


