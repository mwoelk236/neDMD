import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations_with_replacement
plt.rcParams['figure.dpi'] = 150
import time

start_time = time.time()

##################################################################################################################################################################################################
"""Functions"""
#Setting up polynomial dictionary functions for Z 
def polynomial_dictionary(x, degree):
    n_vars = len(x)
    terms = []
    
    # Generate terms: x1, x2, x1^2, x1*x2, x2^2, ...
    for d in range(1, degree + 1):  
        for combo in combinations_with_replacement(range(n_vars), d):
            term = np.prod([x[i] for i in combo])
            terms.append(term)
    
    return np.array(terms)




#############################################################################################################################################################################################
#DATA PRE-PROCESSESING
dt = 0.1 #Time step

#Obtain/Import NOISY DATA
input_folder = f'State Data'
noise_x1 = np.loadtxt(os.path.join(input_folder, 'state_x1.csv'), delimiter = ',')
noise_x2 = np.loadtxt(os.path.join(input_folder, 'state_x2.csv'), delimiter = ',')
noise_x3 = np.loadtxt(os.path.join(input_folder, 'state_x3.csv'), delimiter = ',')
noise_x4 = np.loadtxt(os.path.join(input_folder, 'state_x4.csv'), delimiter = ',')

#Shifted NOISY x1 and x2 
original_noise_x1 = noise_x1[:-1]
original_noise_x2 =  noise_x2[:-1]
original_noise_x3 = noise_x3[:-1]
original_noise_x4 = noise_x4[:-1]

shifted_noise_x1 = noise_x1[1:]
shifted_noise_x2 = noise_x2[1:]
shifted_noise_x3 = noise_x3[1:]
shifted_noise_x4 = noise_x4[1:]

#Setting up Cross-validation Dataset
total_samples = original_noise_x1.shape[1]
train = int(0.8*total_samples)
validation = total_samples - train

train_x1 = original_noise_x1[:,:train]
train_x2 = original_noise_x2[:,:train]  
train_x3 = original_noise_x3[:,:train]
train_x4 = original_noise_x4[:,:train]

val_x1 = original_noise_x1[:,train:]
val_x2 = original_noise_x2[:,train:]
val_x3 = original_noise_x3[:,train:]
val_x4 = original_noise_x4[:,train:]

Train_stacked_X_matrix = np.column_stack((train_x1,train_x2,train_x3,train_x4))
Val_stacked_X_matrix = np.column_stack((val_x1,val_x2,val_x3,val_x4))

shifted_train_x1 = shifted_noise_x1[:,:train]
shifted_train_x2 = shifted_noise_x2[:,:train]
shifted_train_x3 = shifted_noise_x3[:,:train]
shifted_train_x4 = shifted_noise_x4[:,:train]

shifted_val_x1 = shifted_noise_x1[:,train:]
shifted_val_x2 = shifted_noise_x2[:,train:]
shifted_val_x3 = shifted_noise_x3[:,train:]
shifted_val_x4 = shifted_noise_x4[:,train:]

Train_stacked_X_shifted_matrix = np.column_stack((shifted_train_x1,shifted_train_x2,shifted_train_x3,shifted_train_x4))
Val_stacked_X_shifted_matrix = np.column_stack((shifted_val_x1,shifted_val_x2,shifted_val_x3,shifted_val_x4))

#Export and Save the data
output_folder3 = f'Training_Data'
os.makedirs(output_folder3, exist_ok=True)

np.savetxt(os.path.join(output_folder3, 'x1_train.csv'), train_x1, delimiter=',')
np.savetxt(os.path.join(output_folder3, 'x2_train.csv'), train_x2, delimiter=',')
np.savetxt(os.path.join(output_folder3, 'x3_train.csv'), train_x3, delimiter=',')
np.savetxt(os.path.join(output_folder3, 'x4_train.csv'), train_x4, delimiter=',')
np.savetxt(os.path.join(output_folder3, 'shift_x1_train.csv'), shifted_train_x1, delimiter=',')
np.savetxt(os.path.join(output_folder3, 'shift_x2_train.csv'), shifted_train_x2, delimiter=',')
np.savetxt(os.path.join(output_folder3, 'shift_x3_train.csv'), shifted_train_x3, delimiter=',')
np.savetxt(os.path.join(output_folder3, 'shift_x4_train.csv'), shifted_train_x4, delimiter=',')



output_folder4 = f'Validation_Data'
os.makedirs(output_folder4, exist_ok=True)

np.savetxt(os.path.join(output_folder4, 'x1_val.csv'), val_x1, delimiter=',')
np.savetxt(os.path.join(output_folder4, 'x2_val.csv'), val_x2, delimiter=',')
np.savetxt(os.path.join(output_folder4, 'x3_val.csv'), val_x3, delimiter=',')
np.savetxt(os.path.join(output_folder4, 'x4_val.csv'), val_x4, delimiter=',')
np.savetxt(os.path.join(output_folder4, 'shift_x1_val.csv'), shifted_val_x1, delimiter=',')
np.savetxt(os.path.join(output_folder4, 'shift_x2_val.csv'), shifted_val_x2, delimiter=',')
np.savetxt(os.path.join(output_folder4, 'shift_x3_val.csv'), shifted_val_x3, delimiter=',')
np.savetxt(os.path.join(output_folder4, 'shift_x4_val.csv'), shifted_val_x4, delimiter=',')


##############################################################################################################################################################################################
z_dict_degree = 5

#SETTING UP POLYNOMIAL DICTIONARY FUNCTIONS
n_time_steps = train_x1.shape[0]
n_trials = train_x1.shape[1]
z_matrix = []
z_prime_matrix = []


#FOR TRAINING DATA!
for trial in range(n_trials):
    trial_dictionary = []
    prime_trial_dictionary = []

    for time_step in range(n_time_steps):
        trial_state = np.array([train_x1[time_step, trial], train_x2[time_step, trial], train_x3[time_step, trial], train_x4[time_step, trial]])
        prime_trial_state = np.array([shifted_train_x1[time_step,trial], shifted_train_x2[time_step, trial], shifted_train_x3[time_step, trial], shifted_train_x4[time_step, trial]])

        trial_dict = polynomial_dictionary(trial_state, z_dict_degree)
        trial_dictionary.append(trial_dict)

        prime_trial_dict = polynomial_dictionary(prime_trial_state, z_dict_degree)
        prime_trial_dictionary.append(prime_trial_dict)
    
    z_matrix.append(np.stack(trial_dictionary))
    z_prime_matrix.append(np.stack(prime_trial_dictionary))

z_matrix = np.vstack(z_matrix)
z_prime_matrix = np.vstack(z_prime_matrix)

output_folder = f'Z_Data'
os.makedirs(output_folder, exist_ok=True)
np.savetxt(os.path.join(output_folder, 'z.csv'), z_matrix, delimiter=',')
np.savetxt(os.path.join(output_folder, 'z_prime.csv'), z_prime_matrix, delimiter=',')

##################################################################################################################################################################################################
#For Validation Data!
#SETTING UP POLYNOMIAL DICTIONARY FUNCTIONS
val_n_time_steps = val_x1.shape[0]
val_n_trials = val_x1.shape[1]
val_z_matrix = []
val_z_prime_matrix = []

for trial in range(val_n_trials):
    trial_dictionary = []
    prime_trial_dictionary = []

    for time_step in range(val_n_time_steps):
        trial_state = np.array([val_x1[time_step, trial], val_x2[time_step, trial], val_x3[time_step, trial], val_x4[time_step, trial]])
        prime_trial_state = np.array([shifted_val_x1[time_step,trial], shifted_val_x2[time_step, trial], shifted_val_x3[time_step, trial], shifted_val_x4[time_step, trial]])

        trial_dict = polynomial_dictionary(trial_state, z_dict_degree)
        trial_dictionary.append(trial_dict)

        prime_trial_dict = polynomial_dictionary(prime_trial_state, z_dict_degree)
        prime_trial_dictionary.append(prime_trial_dict)
    
    val_z_matrix.append(np.stack(trial_dictionary))
    val_z_prime_matrix.append(np.stack(prime_trial_dictionary))

val_z_matrix = np.vstack(val_z_matrix)
val_z_prime_matrix = np.vstack(val_z_prime_matrix)

#Export and save the data
output_folder2 = f'Validation_Z_Data'
os.makedirs(output_folder2, exist_ok=True)

np.savetxt(os.path.join(output_folder2, 'val_z.csv'), val_z_matrix, delimiter=',')
np.savetxt(os.path.join(output_folder2, 'val_z_prime.csv'), val_z_prime_matrix, delimiter=',')
