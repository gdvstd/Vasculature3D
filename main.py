import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PatchDataset
from unet3d import UNet3D
from utils import *
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

def train(args):
    writer = SummaryWriter(comment=args.name)
    torch.autograd.set_detect_anomaly(True)

    # Set your image_list, patch_size, and other parameters
    image_list = [
        ("/dataset/train/kidney_1_dense/images", "/dataset/train/kidney_1_dense/labels"),
        ("/dataset/train/kidney_2/images", "/dataset/train/kidney_2/labels"),
        ("/dataset/train/kidney_3_sparse/images", "/dataset/train/kidney_3_sparse/labels"),
        ("/dataset/train/kidney_3_sparse/images", "/dataset/train/kidney_3_dense/labels")
    ]

    patch_size = (64, 64, 64)

    # Create an instance of your dataset
    dataset = PatchDataset(image_list, patch_size=patch_size)

    # Create DataLoader for batching
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Instantiate your UNet3D model
    model = UNet3D(in_channels=1, out_channels=len(dataset.labels))
    
    if args.continue_path:
        model.load_state_dict(args.continue_path)

    # Define your loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 100
    batch_idx=0
    print("train start!")
    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            inputs, targets = batch
            
            outputs = model(inputs)
            outputs = outputs.transpose(1, 4)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            #breakpoint()
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('loss', loss.item(), batch_idx)
            print('loss:', loss.item())
            batch_idx += 1

    # Save your model, if needed
    checkpoint = {
        'batch_idx': batch_idx, 
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, f'./save/unet3d_{args.name}.pth')

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('-n', '--name', type=str, help='experiment name')
    parser.add_argument('-c', '--continue_path', default=None, help='path of model checkpoint to continue training')
    
    return  parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    train(args)