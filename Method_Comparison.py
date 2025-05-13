import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
plt.rcParams['figure.dpi'] = 150
import time

start_time = time.time()

##################################################################################################################################################################################################
#FUNCTIONS
def polynomial_basis_noise(ej, degree):
    n_samples = len(ej)
    
    basis_matrix = [np.ones(n_samples)]
    basis_matrix.append(ej)
    
    for d in range(1, degree):
        basis_matrix.append(ej * np.tanh(ej) ** d)
    
    return np.column_stack(basis_matrix)


def diff_polynomial_basis_vectors(ej, degree):
    n_samples = len(ej)
    d_ej_terms = []
    d_ej_terms.append(np.zeros(n_samples))
    d_ej_terms.append(np.ones(n_samples))
    
    # Generate derivatives for the x tanh^d(x) terms
    for d in range(1, degree):
        term_derivative = np.tanh(ej) ** d + ej * d * np.tanh(ej) ** (d - 1) * (1/ np.cosh(ej)** 2)
        d_ej_terms.append(term_derivative)
    
    d_ej_matrix = np.column_stack(d_ej_terms)
    
    return d_ej_matrix


def generate_epsilon_j(N, p):

    if p < 0 or p >= N:
        raise ValueError(f"Index p ({p}) must be in the range [0, {N - 1}]")
    
    epsilon_j = np.zeros((N, 1))
    epsilon_j[p, 0] = 1.0

    return epsilon_j


def obj_function_calc(e_j_1,diff_Z,beta):
    ell = (-e_j_1.shape[0]/2)*np.log(2*np.pi) - (0.5) * np.sum(e_j_1**2)
    J = (-ell/e_j_1.shape[0]) + (beta/2) *np.sum(diff_Z**2)
    return J


def MHE_obj_function_calc(diff_Z):                                                 
    J_MHE = np.sum(diff_Z**2)
    return J_MHE



def plot_trials_separate(val_z_prime_matrix, eDMD_z_prime_matrix,z_prime_MHE, num_trials=None):
    total_trials = val_z_prime_matrix.shape[0] // 600
    if num_trials is None or num_trials > total_trials:
        num_trials = total_trials

    # Initialize lists to store differences
    eDMD_diff_x1 = []
    MHE_diff_x1 = []

    eDMD_diff_x2 = []
    MHE_diff_x2 = []

    eDMD_diff_x3 = []
    solo_diff_x3 = []
    MHE_diff_x3 = []

    eDMD_diff_x4 = []
    MHE_diff_x4 = []

    # Calculate differences for each trial
    for trial in range(num_trials):
        start_idx = trial * 600
        end_idx = start_idx + 600
        eDMD_diff_x1.append(eDMD_z_prime_matrix[start_idx:end_idx, 0] - val_z_prime_matrix[start_idx:end_idx, 0])
        MHE_diff_x1.append(z_prime_MHE[start_idx:end_idx, 0] - val_z_prime_matrix[start_idx:end_idx, 0])

        eDMD_diff_x2.append(eDMD_z_prime_matrix[start_idx:end_idx, 1] - val_z_prime_matrix[start_idx:end_idx, 1])
        MHE_diff_x2.append(z_prime_MHE[start_idx:end_idx, 1] - val_z_prime_matrix[start_idx:end_idx, 1])

        eDMD_diff_x3.append(eDMD_z_prime_matrix[start_idx:end_idx, 2] - val_z_prime_matrix[start_idx:end_idx, 2])
        MHE_diff_x3.append(z_prime_MHE[start_idx:end_idx, 2] - val_z_prime_matrix[start_idx:end_idx, 2])
        
        eDMD_diff_x4.append(eDMD_z_prime_matrix[start_idx:end_idx, 3] - val_z_prime_matrix[start_idx:end_idx, 3])
        MHE_diff_x4.append(z_prime_MHE[start_idx:end_idx, 3] - val_z_prime_matrix[start_idx:end_idx, 3])

    eDMD_diff_x1 = np.array(eDMD_diff_x1)
    solo_diff_x1 = np.array(solo_diff_x1)
    MHE_diff_x1 = np.array(MHE_diff_x1)

    eDMD_diff_x2 = np.array(eDMD_diff_x2)
    solo_diff_x2 = np.array(solo_diff_x2)
    MHE_diff_x2 = np.array(MHE_diff_x2)

    eDMD_diff_x3 = np.array(eDMD_diff_x3)
    solo_diff_x3 = np.array(solo_diff_x3)
    MHE_diff_x3 = np.array(MHE_diff_x3)

    eDMD_diff_x4 = np.array(eDMD_diff_x4)
    solo_diff_x4 = np.array(solo_diff_x4)
    MHE_diff_x4 = np.array(MHE_diff_x4)

    time = np.linspace(0, 60.0, 600)

    # Plot shaded regions for x1
    plt.figure(figsize=(10, 6))
    plt.fill_between(time, np.min(eDMD_diff_x1, axis=0), np.max(eDMD_diff_x1, axis=0), color='b', alpha=0.3, label='eDMD')
    plt.fill_between(time, np.min(MHE_diff_x1, axis=0), np.max(MHE_diff_x1, axis=0), color='g', alpha=0.3, label='neDMD-MHE')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x_1-\tilde{x}_1$')
    plt.xlim([0, max(time)])
    plt.ylim([-0.002, 0.002])
    plt.legend()
    output_folder4 = f'Result_Plots_Separate'
    os.makedirs(output_folder4, exist_ok=True)
    plt.savefig(os.path.join(output_folder4, f'x1.png'))
    plt.close()

    # Plot shaded regions for x2
    plt.figure(figsize=(10, 6))
    plt.fill_between(time, np.min(eDMD_diff_x2, axis=0), np.max(eDMD_diff_x2, axis=0), color='b', alpha=0.3, label='eDMD')
    plt.fill_between(time, np.min(MHE_diff_x2, axis=0), np.max(MHE_diff_x2, axis=0), color='g', alpha=0.3, label='neDMD-MHE')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x_2-\tilde{x}_2$')
    plt.xlim([0, max(time)])
    plt.ylim([-0.05, 0.05])
    plt.legend()
    plt.savefig(os.path.join(output_folder4, f'x2.png'))
    plt.close()

    # Plot shaded regions for x3
    plt.figure(figsize=(10, 6))
    plt.fill_between(time, np.min(eDMD_diff_x3, axis=0), np.max(eDMD_diff_x3, axis=0), color='b', alpha=0.3, label='eDMD')
    plt.fill_between(time, np.min(MHE_diff_x3, axis=0), np.max(MHE_diff_x3, axis=0), color='g', alpha=0.3, label='neDMD-MHE')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x_3-\tilde{x}_3$')
    plt.xlim([0, max(time)])
    plt.ylim([-0.15, 0.15])
    plt.legend()
    plt.savefig(os.path.join(output_folder4, f'x3.png'))
    plt.close()

    # Plot shaded regions for x4
    plt.figure(figsize=(10, 6))
    plt.fill_between(time, np.min(eDMD_diff_x4, axis=0), np.max(eDMD_diff_x4, axis=0), color='b', alpha=0.3, label='eDMD')
    plt.fill_between(time, np.min(MHE_diff_x4, axis=0), np.max(MHE_diff_x4, axis=0), color='g', alpha=0.3, label='neDMD-MHE')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x_4-\tilde{x}_4$')
    plt.xlim([0, max(time)])
    plt.ylim([-0.01, 0.01])
    plt.legend()
    plt.savefig(os.path.join(output_folder4, f'x4.png'))
    plt.close()

#############################################################################################################################################################################################
#Data Loading 

train_z_matrix = np.loadtxt(os.path.join(f'Z_Data', 'z.csv'), delimiter = ',')
train_z_prime_matrix = np.loadtxt(os.path.join(f'Z_Data', 'z_prime.csv'), delimiter = ',')

#Setting Up Dictionary Functions for VALIDATION Data
input_folder = f'Validation_Z_Data'
val_z_matrix = np.loadtxt(os.path.join(input_folder, 'val_z.csv'), delimiter = ',')
val_z_prime_matrix = np.loadtxt(os.path.join(input_folder, 'val_z_prime.csv'), delimiter = ',')



#############################################################################################################################################################################################
#Obtaining A using generic eDMD
eDMD_AT_matrix = np.matmul(np.linalg.pinv(train_z_matrix), train_z_prime_matrix)

#Calculating the error in A matrix
eDMD_z_prime_matrix = np.matmul(val_z_matrix, eDMD_AT_matrix)
Test_RMSE_eDMD = np.sqrt(mean_squared_error(val_z_prime_matrix, np.matmul(val_z_matrix, eDMD_AT_matrix)))

print(f'Test RMSE for eDMD A matrix: {Test_RMSE_eDMD}')


#############################################################################################################################################################################################
#Loading Optimized A Matrix ONLY
input_folder3 = f'AB'
AT_matrix = np.loadtxt(os.path.join(input_folder3, 'A.csv'), delimiter = ',')
BT_matrix = np.loadtxt(os.path.join(input_folder3, 'B.csv'), delimiter = ',')
e1 = np.loadtxt(os.path.join(input_folder3, 'e1.csv'), delimiter = ',')

#############################################################################################################################################################################################
input_folder6 = f'MHE_ej_data'
MHE_e1 = np.loadtxt(os.path.join(input_folder6, 'e1.csv'), delimiter = ',')

print(f"Min:{np.min(MHE_e1)}, and Max:{np.max(MHE_e1)}")

psi_basis_degree = 5
MHE_Psi = polynomial_basis_noise(MHE_e1, psi_basis_degree)
z_prime_MHE = np.matmul(val_z_matrix, AT_matrix) + np.matmul(MHE_Psi, BT_matrix)

Test_RMSE_OUR = np.sqrt(mean_squared_error(val_z_prime_matrix, z_prime_MHE))
print(f'Test RMSE for  MHE: {Test_RMSE_OUR}')


#############################################################################################################################################################################################
#Plotting the results all on one plot
num_trials = 5
plot_trials_separate(val_z_prime_matrix, eDMD_z_prime_matrix, z_prime_MHE, num_trials=num_trials)




