import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
plt.rcParams['figure.dpi'] = 150

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

def obj_function_calc(e_j_1,diff_Z,beta):
    ell = (-e_j_1.shape[0]/2)*np.log(2*np.pi) - (0.5) * np.sum(e_j_1**2)
    J = (-ell/e_j_1.shape[0]) + (beta/2) *np.sum(diff_Z**2)
    return J 



##############################################################################################################################################################################################
#Obtain/Import NOISY DATA
input_folder = f'Z_Data'
z_matrix = np.loadtxt(os.path.join(input_folder, 'z.csv'), delimiter = ',')
z_prime_matrix = np.loadtxt(os.path.join(input_folder, 'z_prime.csv'), delimiter = ',')

# ###############################################################################################################################################################################################
#SETTING UP POLYNOMIAL BASIS FOR NOISE COMPONENTS e1,e2
#Initialize e_j
rows = z_matrix.shape[0]
init_ej = 0.1
e_j_1 = np.random.uniform(-init_ej, init_ej, size=(rows,))

while np.any(e_j_1 == 0):
    e_j_1 = np.random.uniform(-init_ej, init_ej, size=(rows,))

#Setting the degree for Polynomial Basis of e_j
psi_basis_degree = 5
iteration = 0
overall_iteration = 1

# Hyperparameters for gradient descent
eta = 1e0 # Learning Rate/Step Size 
beta = 1e-2  # Weight for the regularization term
tolerance = 1e-3  # Convergence tolerance 
RMSE = 0.03  # Root Mean Squared Error
diff_tol = 1e-4

c = 1e-1  # Armijo condition parameter
rho = 0.9  # Step size reduction factor
max_eta_iterations = 5  # Maximum number of iterations for step size search

outer_counter = 0
max_outer_iterations = 500  # Maximum number of outer loop iterations

all_results_J = []
all_results_J_new = []
all_results_gradient = []
trial_boundaries = []
rmse_values = []

while RMSE >= 1e-3 and outer_counter < max_outer_iterations:
    max_iterations = 1000
    gradient_norm = 1.0
    counter = 0
    results = []
    results_new = []
    results_gradient = []
    J_differences = [] 
    J_new_values = []
    gradient_norms = []

    Psi = polynomial_basis_noise(e_j_1, psi_basis_degree)
    stack_z_psi = np.hstack((z_matrix, Psi))
    MPinverse_z_psi = np.linalg.pinv(stack_z_psi)
    Combined_AB = np.matmul(MPinverse_z_psi, z_prime_matrix)

    # Compute First Block of Gradient Descent
    AT_matrix = Combined_AB[:z_prime_matrix.shape[1], :z_prime_matrix.shape[1]]
    BT_matrix = Combined_AB[z_prime_matrix.shape[1]:, :z_prime_matrix.shape[1]]


    if outer_counter > 0: #Storing Matrices for each outer loop iteration
        output_folder2 = f'AB'
        os.makedirs(output_folder2, exist_ok=True)
        np.savetxt(os.path.join(output_folder2, 'A.csv'), AT_matrix, delimiter=',')
        np.savetxt(os.path.join(output_folder2, 'B.csv'), BT_matrix, delimiter=',')
        np.savetxt(os.path.join(output_folder2, 'e1.csv'), e_j_1, delimiter=',')

    while gradient_norm >= tolerance and counter < max_iterations:
        e1_gradient_vec = np.full((rows,), 0.0)

        Psi = polynomial_basis_noise(e_j_1, psi_basis_degree)

        diff_Z = z_prime_matrix - (np.matmul(z_matrix, AT_matrix) + np.matmul(Psi, BT_matrix))
        BT_diff_Z = np.matmul(BT_matrix, diff_Z.T)
        d_ej1 = diff_polynomial_basis_vectors(e_j_1, psi_basis_degree)

        # Calculating the gradient for each e_j
        for p in range(rows):    
            Jacob_BT_diff_Z1 = np.matmul(d_ej1[p, :], BT_diff_Z)
            epsilon_j = generate_epsilon_j(rows, p)
            e1_gradient_vec[p] = (e_j_1[p] / rows) - (beta * np.matmul(Jacob_BT_diff_Z1, epsilon_j)).item()
            

        e_j_new1 = e_j_1 - eta * e1_gradient_vec
        

        # Calculate new Psi, diff_Z, and objective function value
        Psi_new = polynomial_basis_noise(e_j_new1,  psi_basis_degree)
        diff_Z_new = z_prime_matrix - (np.matmul(z_matrix, AT_matrix) + np.matmul(Psi_new, BT_matrix))
        J_current = obj_function_calc(e_j_1,  diff_Z, beta) 
        J_new = obj_function_calc(e_j_new1,  diff_Z_new, beta)

        e_j_1 = e_j_new1
        
        gradient_norm = np.linalg.norm(e1_gradient_vec)


        # Armijo backtracking for optimal step size
        for _ in range(max_eta_iterations):
            e_j_arm1 = e_j_1 - eta * e1_gradient_vec

            Psi_arm = polynomial_basis_noise(e_j_arm1, psi_basis_degree)
            diff_Z_arm = z_prime_matrix - (np.matmul(z_matrix, AT_matrix) + np.matmul(Psi_arm, BT_matrix))
            J_arm = obj_function_calc(e_j_arm1, diff_Z_arm, beta)

            # Check the Armijo condition
            if J_arm <= J_new - c * eta * gradient_norm**2:
                break  # Condition satisfied
            eta *= rho  # Reduce the step size

        if counter % 100 == 0:
            print(f"Outer Iteration {outer_counter}, Iteration {counter}: Gradient Norm = {gradient_norm}, J_new = {J_new}, J_diff = {J_current - J_new}, eta = {eta}")

        J_differences.append(J_current - J_new)
        J_new_values.append(J_new)
        gradient_norms.append(gradient_norm)
        counter += 1

        #Imposed constraint to speed up optimization 
        obj_func_diff = np.abs(J_new-J_current)
        if obj_func_diff <= diff_tol:
            break

    results.append(J_differences)
    results_new.append(J_new_values)
    results_gradient.append(gradient_norms)

    results = np.array(results)
    results_new = np.array(results_new)
    results_gradient = np.array(results_gradient)

    all_results_J.extend(results[0])
    all_results_J_new.extend(results_new[0])
    all_results_gradient.extend(results_gradient[0])
    trial_boundaries.append(len(all_results_J))
    rmse_values.append(RMSE)

    estimated_Z = (np.matmul(z_matrix, AT_matrix) + np.matmul(Psi_new, BT_matrix))
    RMSE = np.sqrt(mean_squared_error(z_prime_matrix, estimated_Z))
    print(f"Outer Iteration {outer_counter}: RMSE = {RMSE}")


    outer_counter += 1




#############################################################################################################################################################################
######Plotting the results######

output_folder = f'Trials_PLot'
os.makedirs(output_folder, exist_ok=True)
plt.savefig(os.path.join(output_folder, f'combined_plot_{outer_counter}.png'))
plt.figure(figsize=(10, 15))

# Plot J vs iteration number for all trials
plt.subplot(3, 1, 1)
plt.plot(all_results_J, label="$J$")
for boundary in trial_boundaries:
    plt.axvline(x=boundary, color='r', linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("$J_{\\text{current}}-J_{\\text{new}}$")
plt.legend()
plt.grid()

# Plot J_new vs iteration number for all trials
plt.subplot(3, 1, 2)
plt.plot(all_results_J_new, label="$J_{\\text{new}}$")
for boundary in trial_boundaries:
    plt.axvline(x=boundary, color='r', linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("$J_{\\text{new}}$")
plt.legend()
plt.grid()

# Plot gradient norm vs iteration number for all trials
plt.subplot(3, 1, 3)
plt.plot(all_results_gradient, label="Gradient Norm")
for boundary in trial_boundaries:
    plt.axvline(x=boundary, color='r', linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, 'combined_plot_all_trials.png'))

# Plot RMSE vs outer iteration number
plt.figure(figsize=(10, 5))
plt.plot(range(len(rmse_values)), rmse_values, marker='o', label="RMSE")
plt.xlabel("Outer Iteration")
plt.ylabel("RMSE")
plt.legend()
plt.grid()

plt.savefig(os.path.join(output_folder, 'rmse_plot.png'))
