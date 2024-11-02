# File: logging.py
# Author: Ethan Ondzik
# Last Changed: Nov 1, 2024

import datetime
import os
import getpass

def write_to_log():
    log_file_name = 'log/run_model.txt'
    
    #with f open('log/run_model.txt')

    date = datetime.datetime.now()
    date_string = date.strftime('%Y/%m/%d %H:%M:%S')
    user_string = getpass.getuser() #should work for most OS
    
    #assuming hyperparameters are global
    base_exponent = 5
    filters = [2 ** (base_exponent + i) for i in range(5)]
    he_uniform = 'he_uniform'


    hyperparameters = {
        'input_shape': (512, 512, 3),
        'filters': filters,
        'kernel_size': (3, 3),
        'activation': 'relu',
        'padding': 'same',
        # 'initializer': he_uniform(),
        'initializer': he_uniform,
        'optimizer': 'adam',
        # 'loss': weightedBinaryCrossEntropy,
        'loss': 'binary_crossentropy',
        'weights' : {0 : 1.0, 1 : 5.0},
        'metrics': ['accuracy'],
        'epochs': 100,
        'batch_size': 4,
        'early_stopping_patience': 10,
        'test_size': 0.2,
        'random_state': 0,
        'seed': 42
    }

    if not os.path.exists('log'):
        os.makedirs('log')
    

    with open(log_file_name, 'a') as log:
        log.write('\n' + date_string + ' ' + user_string + '\n')
        log.write(str(hyperparameters))



if __name__ == '__main__':
    write_to_log()


