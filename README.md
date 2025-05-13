# neDMD
Noisy Koopman Model Identification

## Files Overview

- `RUN.py`: Main script that executes all other files in sequence.
- `Example_dist_binary_sim.py`: Simulator for binary distillation column
- `dictionary_func.py`: Creates the library of dictionary functions and calculates lifting
- `neDMD.py`: Runs the neDMD algorithm to obtain A, B matrices, and noise components
- `MHE.py`: Moving Horizon Estimation file, 
- `Method_Comparison.py`: Performance comparison of vanilla eDMD to neDMD
- 
## Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- os
