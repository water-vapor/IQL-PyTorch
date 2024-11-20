# %%
import torch

def modify_weights(source_file, target_file, coeff):
    a = torch.load(source_file)
    for key, value in a.items():
        a[key] = value * coeff
    torch.save(a, target_file)

def quantize_weights(source_file, target_file):
    a = torch.load(source_file)
    for key, value in a.items():
        a[key] = torch.round(value, decimals=2)
    torch.save(a, target_file)

#%%
folder = 'ref_ckpts_0411/'
relu_files = [
    'relu_umaze',
    'relu_medium',
    'relu_large',
]

for f in relu_files:
    quantize_weights(f"{folder}{f}.pt", f"{folder}{f}_quantized.pt")

for f in relu_files:
    modify_weights(f"{folder}{f}_quantized.pt", f"{folder}{f}_quantized_id.pt", 1.0)
    modify_weights(f"{folder}{f}_quantized.pt", f"{folder}{f}_quantized_para20.pt", 0.936147152)
    modify_weights(f"{folder}{f}_quantized.pt", f"{folder}{f}_quantized_para40.pt", 0.846740635)
    modify_weights(f"{folder}{f}_quantized.pt", f"{folder}{f}_quantized_para60.pt", 0.755842412)
    modify_weights(f"{folder}{f}_quantized.pt", f"{folder}{f}_quantized_para_rel.pt", 0.979870429)
    modify_weights(f"{folder}{f}_quantized.pt", f"{folder}{f}_quantized_perp20.pt", 0.973701916)
    modify_weights(f"{folder}{f}_quantized.pt", f"{folder}{f}_quantized_perp40.pt", 0.824490036)
    modify_weights(f"{folder}{f}_quantized.pt", f"{folder}{f}_quantized_perp60.pt", 0.746702839)
    modify_weights(f"{folder}{f}_quantized.pt", f"{folder}{f}_quantized_perp_rel.pt", 0.90592226)