import os 

scripts = ["dist_binary_sim.py", "dictionary_func.py", "koop.py", "MHE.py", "Method_Comparison.py"]

for script in scripts:
    print(f"Running {script}...")
    os.system(f"python {script}")
    print(f"Finished running {script}\n")