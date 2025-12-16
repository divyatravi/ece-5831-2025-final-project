import torch
from src.model import RobustCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1. Create model
model = RobustCNN().to(device)

# 2. Load saved state dict
state_dict = torch.load("best_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully!")

# 3. Quick sanity check with dummy input
dummy = torch.randn(1, 3, 128, 128).to(device)
with torch.no_grad():
    out = model(dummy)

print("Output shape:", out.shape)  # should be [1, 2]
