import torch
import numpy as np

# Load the model
cut_pth_path='/mnt/32THHD/yw/exp2/output/kitti_models/fade_cutbev_1_1/fade_bevcut_1_1/ckpt/checkpoint_epoch_77.pth'
cut_model = torch.load(cut_pth_path)

pth_path='/mnt/32THHD/yw/exp2/output/kitti_models/fade/baseline_fade_batch_size_32/ckpt/checkpoint_epoch_75.pth'
model = torch.load(pth_path)

for key in model['model_state']:
    if 'dense_head' in key:
        cut_model['model_state'][key]=model['model_state'][key]
        
torch.save(model, '/mnt/32THHD/yw/exp2/cut+fadehead.pth')