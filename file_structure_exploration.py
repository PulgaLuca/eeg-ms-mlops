"""
This is not a structured/production-ready file.
Only some tests to leverage data knowledge from given matlab files.
"""

import scipy.io as sio

# Just a loading test with PSD/ms
data = sio.loadmat(r"C:\Users\lucap\Desktop\Tesi_LucaPulga\eeg-ms-mlops\data\raw\PSD\ms\PSDrelative_ID_01_T0_CE.mat")

print()
for k, v in data.items():
    if not k.startswith("__"):  # Skip some MATLAB metadata
        print(f"{k:25s} {v.shape} {v.dtype}")

'''
28 variables
banda/banda_rel
PSD_tot
ratios: theta/alpha, delta/alpha, SFR
'''
    