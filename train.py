import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from utils.metrics import _dice_loss, evaluation, contrastive_loss
from utils import prepare_dataset
from utils.dataset import LandmarkDataset
from models.D2GPLand import D2GPLand, Edge_Prototypes


def main(save_path, args):
    train_file, test_file, val_file = prepare_dataset.get_split(args.data_path)
    train_transform = T.Compose([
        T.ToTensor(),
        T.RandomResizedCrop(1024, scale=(0.8, 1.2)),
        T.RandomHorizontalFlip(),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
    ])
    train_dataset = LandmarkDataset(train_file, transform=train_transform, mode='train')
    val_dataset = LandmarkDataset(val_file, transform=val_transform, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda")
    bce_loss = torch.nn.BCEWithLogitsLoss()
    cl_loss = contrastive_loss()
    best_dice = -100

    model = D2GPLand(1024, 1024).to(device)
    model.sam_encoder.requires_grad_(False)

    edge_prototypes_model = Edge_Prototypes(num_classes=args.num_landmark, feat_dim=256).to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': edge_prototypes_model.parameters()}
                                  ], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, args.decay_lr)

    for epoch in range(args.epoch):
        epoch_running_loss = 0
        epoch_seg_loss = 0
        epoch_contrastive_loss = 0
        epoch_edge_loss = 0

        # trainng
        model.train()
        edge_prototypes_model.train()
        for batch_idx, (X_batch, depth, y_batch, *rest) in tqdm(enumerate(train_loader)):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            depth = depth.to(device)
            prototypes = edge_prototypes_model()

            output, feature, edge_out = model(X_batch, depth, prototypes)

            prototype_loss = cl_loss(feature, y_batch, prototypes, device)
            edge_gt = F.interpolate(y_batch[:, 1:, :, :], size=16, mode="bilinear")
            edge_loss = _dice_loss(output, y_batch) + bce_loss(output, y_batch)
            seg_loss = _dice_loss(edge_out, edge_gt) + bce_loss(edge_out, edge_gt)
            loss = seg_loss + prototype_loss + edge_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_running_loss += loss.item()
            epoch_seg_loss += seg_loss.item()
            epoch_contrastive_loss += prototype_loss.item()
            epoch_edge_loss += edge_loss.item()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch, args.epoch, epoch_running_loss / (batch_idx + 1)))
        print('epoch [{}/{}], seg loss:{:.4f}'
              .format(epoch, args.epoch, epoch_seg_loss / (batch_idx + 1)))
        print('epoch [{}/{}], contrastive loss:{:.4f}'
              .format(epoch, args.epoch, epoch_contrastive_loss / (batch_idx + 1)))
        print('epoch [{}/{}], edge loss:{:.4f}'
              .format(epoch, args.epoch, epoch_edge_loss / (batch_idx + 1)))

        # validation
        model.eval()
        edge_prototypes_model.eval()
        validation_IOU = []
        mDice = []

        with torch.no_grad():
            for X_batch, depth, y_batch, name in tqdm(val_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                depth = depth.to(device)
                prototypes = edge_prototypes_model()

                output, feature, edge_out = model(X_batch, depth, prototypes)
                output = torch.argmax(torch.softmax(output, dim=1), dim=1)
                y_batch = torch.argmax(y_batch, dim=1)

                tmp2 = y_batch.detach().cpu().numpy()
                tmp = output.detach().cpu().numpy()
                tmp = tmp[0]
                tmp2 = tmp2[0]

                pred = np.array([tmp == i for i in range(4)]).astype(np.uint8)
                gt = np.array([tmp2 == i for i in range(4)]).astype(np.uint8)

                iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())

                validation_IOU.append(iou)
                mDice.append(dice)

        print(np.mean(validation_IOU))
        print(np.mean(mDice))
        if np.mean(mDice) > best_dice:
            best_dice = np.mean(mDice)
            torch.save(model.state_dict(), save_path + "best_model_path.pth")
            torch.save(model.state_dict(), save_path + "best_prototype_path.pth")
        print("best dice is:{:.4f}".format(best_dice))
    scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--weight_decay', default=3e-5)
    parser.add_argument('--decay_lr', default=1e-6)
    parser.add_argument('--epoch', default=60)
    parser.add_argument('--num_landmark', default=3)
    parser.add_argument('--data_path', default='L3D/')
    args = parser.parse_args()

    save_path = 'results/'
    os.makedirs(save_path, exist_ok=True)

    main(save_path, args=args)
