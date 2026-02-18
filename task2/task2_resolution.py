import torch
from torch.utils.data import DataLoader
from models.fno import FNO1d

from config import device, DATA_PATHS, MODEL_PATHS, TASK1_CONFIG
from datasets import One2OneDataset
from utils import compute_relative_l2_error, load_model


print("\nTASK 2: Testing on Different Resolutions")

# ============================================
# LOAD PRETRAINED MODEL FROM TASK 1
# ============================================
print("\nLoading Task 1 model...")
model = FNO1d(
    modes=TASK1_CONFIG['modes'], 
    width=TASK1_CONFIG['width'], 
    in_dim=2, 
    out_dim=1
).to(device)

model = load_model(model, MODEL_PATHS['task1_one2one'], device)

# ============================================
# TEST ON DIFFERENT RESOLUTIONS
# ============================================
resolutions = [32, 64, 96, 128]
errors = {}

print("\nTesting on different resolutions...")

for res in resolutions:
    # Load test dataset at this resolution
    data_path = DATA_PATHS[f'test_{res}']
    test_dataset = One2OneDataset(data_path, TASK1_CONFIG['n_test'], res)
    test_loader = DataLoader(test_dataset, batch_size=TASK1_CONFIG['batch_size'], shuffle=False)
    
    # Evaluate
    model.eval()
    test_error = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_error += compute_relative_l2_error(outputs, targets)
    
    test_error /= TASK1_CONFIG['n_test']
    errors[res] = test_error
    
    print(f'Resolution {res:3d}: Error = {test_error:.6f}')
