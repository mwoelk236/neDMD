import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.dpi'] = 150
import time

start_time = time.time()

##################################################################################################################################################################################################
"""Functions"""
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
        term_derivative = np.tanh(ej) ** d + ej * d * np.tanh(ej) ** (d - 1) * (1/ np.cosh(ej) ** 2)
        d_ej_terms.append(term_derivative)
    
    d_ej_matrix = np.column_stack(d_ej_terms)
    
    return d_ej_matrix


def generate_epsilon_j(N, p):
    if p < 0 or p >= N:
        raise ValueError(f"Index p ({p}) must be in the range [0, {N - 1}]")
    
    epsilon_j = np.zeros((N, 1))
    epsilon_j[p, 0] = 1.0

    return epsilon_j


#Calculating the Objective Function Value J at given e_j
def obj_function_calc(e_j_1,diff_Z,beta):
    ell = (-e_j_1.shape[0]/2)*np.log(2*np.pi) - (0.5) * np.sum(e_j_1**2 )
    J = (-ell/e_j_1.shape[0]) + (beta/2) *np.sum(diff_Z**2)
    return J


#Calculating the Objective Function Value for MHE J at given e_j
def MHE_obj_function_calc(accum_error):
    J_MHE = 0.0
    total_square_row_sum = 0.0
    for i in range(accum_error.shape[0]):
        for j in range(accum_error.shape[1]):
            square_row_sum = (accum_error[i,j])**2
            total_square_row_sum =  square_row_sum + total_square_row_sum

        J_MHE = J_MHE + total_square_row_sum
        total_square_row_sum = 0.0

    return J_MHE

#############################################################################################################################################################################################
#Obtain/Import NOISY DATA
input_folder = f'Z_Data'
train_z_matrix = np.loadtxt(os.path.join(input_folder, 'z.csv'), delimiter = ',')
train_z_prime_matrix = np.loadtxt(os.path.join(input_folder, 'z_prime.csv'), delimiter = ',')

#Setting Up Dictionary Functions for VALIDATION Data
input_folder = f'Validation_Z_Data'
val_z_matrix = np.loadtxt(os.path.join(input_folder, 'val_z.csv'), delimiter = ',')
val_z_prime_matrix = np.loadtxt( os.path.join(input_folder, 'val_z_prime.csv'), delimiter = ',')

#############################################################################################################################################################################################
"""Moving Horizon Estimation """
storage_e1 = []

#Loading A and B matrices
input_folder3 = f'AB'
AT_matrix = np.loadtxt(os.path.join(input_folder3, 'A.csv'), delimiter = ',')
BT_matrix = np.loadtxt(os.path.join(input_folder3, 'B.csv'), delimiter = ',')
num_trials = 10
horizon_init = 5
psi_basis_degree = 5

for y in range(num_trials):
    trial_length = 600
    length = y *trial_length

    horizon_val_z_matrix = val_z_matrix[length:length+horizon_init, :]
    horizon_val_z_prime_matrix = val_z_prime_matrix[length:length+horizon_init, :]

    rows = horizon_val_z_matrix.shape[0]

    init_ej = 0.01
    horizon_e1 = np.random.uniform(-init_ej, init_ej, size=(rows,))
    
    while np.any(horizon_e1 == 0):
        horizon_e1 = np.random.uniform(-init_ej, init_ej, size=(rows,))


    iteration = 0
    overall_iteration = 1

    # Hyperparameters for gradient descent
    eta = 1.0  # Learning Rate/Step Size 
    beta = 1e-2  # Weight for the regularization term
    tolerance = 1e-4  # Convergence tolerance 
    RMSE = 0.03  # Root Mean Squared Error

    c = 1e-1  # Armijo condition parameter
    rho = 0.9  # Step size reduction factor
    max_eta_iterations = 5  # Maximum number of iterations for step size search
    max_iterations = 500
    gradient_norm = 1.0
    counter = 0

    while gradient_norm >= tolerance and counter < max_iterations:
        e1_gradient_vec = np.full((rows,), 0.0)

        Psi = polynomial_basis_noise(horizon_e1, psi_basis_degree)

        diff_Z = horizon_val_z_prime_matrix - (np.matmul(horizon_val_z_matrix, AT_matrix) + np.matmul(Psi, BT_matrix))
        BT_diff_Z = np.matmul(BT_matrix, diff_Z.T)
        d_ej1 = diff_polynomial_basis_vectors(horizon_e1, psi_basis_degree)

        # Calculating the gradient for each e_j
        for p in range(rows):    
            Jacob_BT_diff_Z1 = np.matmul(d_ej1[p, :], BT_diff_Z)

            epsilon_j = generate_epsilon_j(rows, p)

            e1_gradient_vec[p] = (horizon_e1[p] / rows) - (beta * np.matmul(Jacob_BT_diff_Z1, epsilon_j)).item()

        e_j_new1 = horizon_e1 - eta * e1_gradient_vec

        # Calculate new Psi, diff_Z, and objective function value
        Psi_new = polynomial_basis_noise(e_j_new1, psi_basis_degree)
        diff_Z_new = horizon_val_z_prime_matrix - (np.matmul(horizon_val_z_matrix, AT_matrix) + np.matmul(Psi_new, BT_matrix))
        J_current = obj_function_calc(horizon_e1, diff_Z, beta) 
        J_new = obj_function_calc(e_j_new1, diff_Z_new, beta)

        horizon_e1 = e_j_new1

        gradient_norm = np.linalg.norm(e1_gradient_vec)

        # Armijo backtracking for optimal step size
        for _ in range(max_eta_iterations):
            # Update e_j values
            e_j_arm1 = horizon_e1 - eta * e1_gradient_vec

            Psi_arm = polynomial_basis_noise(e_j_arm1, psi_basis_degree)
            diff_Z_arm = horizon_val_z_prime_matrix - (np.matmul(horizon_val_z_matrix, AT_matrix) + np.matmul(Psi_arm, BT_matrix))
            J_arm = obj_function_calc(e_j_arm1, diff_Z_arm, beta)
            # Check the Armijo condition
            if J_arm <= J_new - c * eta * gradient_norm**2:
                break  
            eta *= rho  # Reduce the step size
     
        counter +=1

    for i in range(horizon_e1.shape[0]):
        storage_e1.append(horizon_e1[i])
        

################################################################################################################################################################################
#Optimize the e_j values for the times past horizon length 

Horizon_Length = 5
horizon_start = rows #This should be 5 based on above
MHE_init_ej = 0.0
MHE_grad_tol = 1e-5
MHE_max_iter = 1000
MHE_Eta = 1e-3

init_MHE_e1 = np.full((Horizon_Length,),MHE_init_ej)

window_counter = 0
max_idx = Horizon_Length

#while horizon_start > val_z_matrix.shape[0] - 1 - Horizon_Length:
while horizon_start < trial_length-4:

    if window_counter > 0:
        init_MHE_e1[0:-1] = MHE_e1[1:]

    current_horizon_start = horizon_start 

    horizon_z_matrix = val_z_matrix[current_horizon_start+length:length+current_horizon_start+Horizon_Length, :] 
    horizon_z_prime_matrix = val_z_prime_matrix[current_horizon_start+length:length+current_horizon_start + Horizon_Length, :]

    init_Psi = polynomial_basis_noise(init_MHE_e1, psi_basis_degree)
    init_dJ_ej1 = diff_polynomial_basis_vectors(init_MHE_e1, psi_basis_degree)

    z_k_pred = np.matmul(horizon_z_matrix, AT_matrix) + np.matmul(init_Psi, BT_matrix)
    z_k1 = horizon_z_prime_matrix

    error = z_k1 - z_k_pred
    J_MHE = MHE_obj_function_calc(error)

    #Now Update the e_j values
    iter = 0
    MHE_grad_norm = 1.0
    while MHE_grad_norm >= MHE_grad_tol and iter < MHE_max_iter: 

        if iter == 0:
            MHE_e1 = init_MHE_e1

        MHE_grad_e1 = np.full((Horizon_Length,), 0.0)

        MHE_Psi = polynomial_basis_noise(MHE_e1, psi_basis_degree)
        MHE_d_ej1 = diff_polynomial_basis_vectors(MHE_e1, psi_basis_degree)
        
        MHE_z_k_pred = np.matmul(horizon_z_matrix, AT_matrix) + np.matmul(MHE_Psi, BT_matrix)

        for p in range(Horizon_Length):
            epsilon_j_MHE = generate_epsilon_j(Horizon_Length,p)

            MHE_grad_e1[p] = -2 * np.matmul(np.matmul(MHE_d_ej1[p, :], np.matmul(BT_matrix, (z_k1 - MHE_z_k_pred ).T)), epsilon_j_MHE).item()

        MHE_e1_new = MHE_e1 - MHE_Eta * MHE_grad_e1

        #Calculating updated J_MHE objective function value 
        MHE_Psi_new = polynomial_basis_noise(MHE_e1_new, psi_basis_degree)
        MHE_z_k_pred_new = np.matmul(horizon_z_matrix, AT_matrix) + np.matmul(MHE_Psi_new, BT_matrix)

        new_error = z_k1 - MHE_z_k_pred_new

        new_J_MHE = MHE_obj_function_calc(new_error)

        MHE_e1 = MHE_e1_new

        MHE_combined_gradient = np.concatenate((MHE_grad_e1,))
        MHE_grad_norm = np.linalg.norm(MHE_combined_gradient)

        iter += 1

    storage_e1.append(MHE_e1[0])

    if horizon_start == trial_length-5:
        for k in  range(MHE_e1.shape[0]-1):
            storage_e1.append(MHE_e1[k+1])

    horizon_start += 1
    window_counter += 1
print(f"Trial {y+1} Completed")


e1 = np.array(storage_e1)

output_folder45 = f'MHE_ej_data'
os.makedirs(output_folder45, exist_ok=True)

e1_file = os.path.join(output_folder45, 'e1.csv')
np.savetxt(e1_file, e1, delimiter=',')
