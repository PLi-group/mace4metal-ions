#!/usr/bin/env python
import torch

model_file = 'MACE_XXX_v1_stagetwo.model'

# 1. Need to set weights_only=False to load the whole MACE class structure
print(f"Loading model: {model_file}...")
model = torch.load(model_file, weights_only=False)

# 2. Convert the model to float32
print("Converting model to float32...")
model = model.to(torch.float32)

# 3. Save as a new file
fp32_model_file = 'MACE_XXX_v1_stagetwo_FP32.model'
torch.save(model, fp32_model_file)
print(f"Successfully saved FP32 model to: {fp32_model_file}")

quit()

