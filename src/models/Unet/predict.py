import torch
import numpy as np

from utils import create_single_frame_loaders
import os
import sys

current_file_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))
sys.path.insert(0, project_root)

from src.utils.Datasets import DatasetKol, DatasetCylinder

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    print(f'加载检查点，epoch: {checkpoint["epoch"]}, loss: {checkpoint["loss"]:.6f}')
    return checkpoint['epoch'], train_losses, val_losses

def rollout_predict(model, input_frame, prediction_steps, device):
    model.eval()
    model.to(device)

    predictions = [input_frame]

    with torch.no_grad():
        for _ in range(prediction_steps):
            input_frame = model(input_frame)
            predictions.append(input_frame)
            print('predicted No.', _)

    return np.array(predictions)


if __name__ == "__main__":
    from channel_K_residual_all import channel_UNET
    from K_residual_all import UNET
    from residual_con import UNET_InputResidual
    from no_residual import UNET_NoResidual
    
    val_index = 5
    
    # channel_lowepoch, inputres, nores, K_residual_all
    checkpoint_path = "./results/res_test_cyl/channel_K_residual_all/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = channel_UNET(in_channels=2, out_channels=2)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    dataset = DatasetCylinder("data/Cylinder", normalize=True, train_ratio=0.8, random_seed=42)

    val_data = dataset.val_data

    val_sample = val_data[val_index]
    print(val_sample.shape)

    val_sample = torch.from_numpy(val_sample).float().unsqueeze(0).to(device)
    print(val_sample.shape)

    input_frame = val_sample[:, 0, :, :]
    # input_frame = input_frame.unsqueeze(1)
    print(input_frame.shape)

    predictions = rollout_predict(model, input_frame, 99, device)
    print(predictions.shape)
    predictions = predictions.squeeze(1)
    print(predictions.shape)

    np.save("results/res_test_cyl/predictions/channel_pred.npy", predictions)