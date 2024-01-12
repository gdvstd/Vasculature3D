import os
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

    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    stride = (args.patch_size, args.patch_size, args.patch_size)
    random_crop = True
    seed = 42

    # Create an instance of your dataset
    print("load train dataset", end='')
    train_dataset = PatchDataset(image_list, split='train', seed=seed, patch_size=patch_size, stride=stride, random_crop=random_crop)
    print("load validation dataset", end='')
    val_dataset = PatchDataset(image_list, split='test', seed=seed, patch_size=patch_size, stride=stride, random_crop=random_crop)

    # Create DataLoader for batching
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Instantiate your UNet3D model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=1, out_channels=len(train_dataset.labels))
    if args.continue_path:
        model.load_state_dict(args.continue_path)
    model = model.to(device)

    # Define your loss function and optimizer
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 100
    batch_idx=0
    print("train start!")
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader):
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            
            outputs = model(inputs).transpose(1, 4)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            #breakpoint()
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('loss', loss.item(), batch_idx)
            print('loss:', loss.item())

            if batch_idx % args.save_frequency == 0:
                checkpoint = {
                    'batch_idx': batch_idx, 
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, f'./save/unet3d_{args.name}_{batch_idx}.pth')
            batch_idx += 1

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('-n', '--name', type=str, help='experiment name')
    parser.add_argument('-c', '--continue_path', default=None, help='path of model checkpoint to continue training')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-p', '--patch_size', type=int, help='patch size')
    parser.add_argument('-sf', '--save_frequency', type=int, required=True, help='saving step frequency')
    
    return  parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    train(args)