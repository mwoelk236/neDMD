import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
import numpy.random as rd
plt.rcParams['figure.dpi'] = 300


##########################################################################################################################################################################
"""Functions"""
def get_initial_state(x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, x4_min, x4_max):
        x10 = rd.uniform(x1_min, x1_max)
        x20 = rd.uniform(x2_min, x2_max)
        x30 = rd.uniform(x3_min, x3_max)
        x40 = rd.uniform(x4_min, x4_max)
        x0 = [x10, x20, x30, x40]
        return  x0


def system_noise(t_max,dt):
    #Vary the width of the gaussian 
    variance_3 = 0.4
    e1 = np.zeros((int(t_max/dt),1))

    for j in range(int(t_max/dt)):
        e1[j,0] = rd.normal(0,variance_3)
    return e1


#ODE System
def dxdt(t, x):
    #Feed Input 
    z_f = -0.3*np.sin(0.5*np.pi*t*e1[int(t/time_step)-1,0]) + 0.6    #Sinusoidal signal as input which represents the feed composition, for no noise, set e1 = 0
    
    #ODE Equations 
    dx1dt = (-V*x[0])/H1 + (V*beta*x[1])/(H1 * (1+((beta-1)*x[1])))
    dx2dt = (L*(x[0]-x[1]))/H2 + (V/H2)*((beta*x[2])/(1+((beta-1)*x[2])) - (beta*x[1])/(1+((beta-1)*x[1])))
    dx3dt = (-F*x[2])/H3 + (L*(x[1]-x[2]))/H3 +  (V/H3)*((beta*x[3])/(1+((beta-1)*x[3])) - (beta*x[2])/(1+((beta-1)*x[2]))) + (F*(z_f)/H3)
    dx4dt = ((F+L)*(x[2]-x[3]))/H4 + (V/H4)*(x[3] - (beta*x[3])/(1+((beta-1)*x[3])))

    return [dx1dt, dx2dt, dx3dt, dx4dt]
##########################################################################################################################################################################

#Numbers indicate the tray (i=1,2,3,4) where i =1 is the top, and i=3 is the feed tray
#Constants
V = 6.05 #Vapor flowrate [=] mol/min
F = 1.70 #Feed flowrate [=] mol/min
L = 4.79 #Reflux flowrate [=] mol/min
beta = 1.60 #Relative volativity constant 

#Time Setup [=] min
t_start = 0.0
t_end = 60.0
time_step = 0.1  # Time step, tau
t_eval = np.arange(t_start, t_end + time_step, time_step) 
t_span = (t_start, t_end)

n_traj = 20 #Number of trajectories to simulate

H1 = 5.25 #Liquid holdup x1 [=] mol
H2 = 0.53 #Liquid holdup x2 [=] mol
H3 = 0.53 #Liquid holdup x3 [=] mol
H4 = 5.25#Liquid holdup x4 [=] mol

x1 = np.zeros((int(t_end/time_step)+1,n_traj))
x2 = np.zeros((int(t_end/time_step)+1,n_traj))
x3 = np.zeros((int(t_end/time_step)+1,n_traj))
x4 = np.zeros((int(t_end/time_step)+1,n_traj))

#Set Initial ranges for s,ul for random initial conditions
x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, x4_min, x4_max = 0.9, 0.95, 0.90, 0.86, 0.81, 0.85, 0.70, 0.80
x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, x4_min, x4_max = 0.88, 0.92, 0.84, 0.88, 0.83, 0.86, 0.72, 0.78

for i in range(n_traj):
    e1 = system_noise(t_end,time_step)
    x0 = get_initial_state(x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, x4_min, x4_max) #Obtaining initial conditions for each trial 
    sol = solve_ivp(dxdt, t_span, x0, method="RK45", t_eval=t_eval)
    t = sol.t
    x = sol.y

    x1[:,i] = x[0,:].T
    x2[:,i] = x[1,:].T
    x3[:,i] = x[2,:].T
    x4[:,i] = x[3,:].T

    output_folder = f'State Data'
    os.makedirs(output_folder, exist_ok=True)

    #Plotting 
    plt.figure()
    plt.plot(t, x[0], label=r"$x_{1}$" )
    plt.plot(t, x[1], label=r"$x_{2}$")
    plt.plot(t, x[2], label=r"$x_{3}$")
    plt.plot(t, x[3], label=r"$x_{4}$")
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$x$', fontsize=18)
    plt.xlim(0,t_end)
    plt.ylim(0.0,1.0)
    plt.legend()
    #plot_filename = f'SystemTrajectory_withNoise_{i}.png'
    #plot_filename = f'SystemTrajectory_withNoise.png'
    plot_filename = f'SystemTrajectory.png'
    plt.savefig(os.path.join(output_folder, plot_filename))


x1_file = os.path.join(output_folder, 'state_x1.csv')
x2_file = os.path.join(output_folder, 'state_x2.csv')
x3_file = os.path.join(output_folder, 'state_x3.csv')
x4_file = os.path.join(output_folder, 'state_x4.csv')

np.savetxt(x1_file, x1, delimiter=',')
np.savetxt(x2_file, x2, delimiter=',')
np.savetxt(x3_file, x3, delimiter=',')
np.savetxt(x4_file, x4, delimiter=',')



