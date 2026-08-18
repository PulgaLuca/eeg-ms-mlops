import scipy.io as sio

data = sio.loadmat(r"C:\Users\lucap\Desktop\Tesi_LucaPulga\data\Features\canali.mat")

# ROI extraction
ROI = {k: v for k, v in data.items() if k.startswith("R")}
print(f"N. ROI: {len(ROI)}")
print("Found ROI:", list(ROI.keys()), "\n")

# Channels extraction
labels = [str(c["labels"]) for c in data["chanlocs"][0]]
print(f"N. lables/channels: {len(labels)}")
print("Found lables/channels:", labels, "\n")

for r in ROI:
    print(r, [str(x[0]) for x in data[r][0]])

'''
R1 ['Fp1', 'F3', 'F7']                      frontale sx
R2 ['Fp2', 'F4', 'F8']                      frontale dx
R3 ['C3','T7','FC1','CP1','FC5','CP5']      centro-temporale sx
R4 ['C4','T8','FC2','CP2','FC6','CP6']      centro-temporale dx
R5 ['P3', 'O1', 'P7']                       parieto-occipitale sx
R6 ['P4', 'O2', 'P8']                       parieto-occipitale dx
'''
