import os
import torch
from tqdm import tqdm
import numpy as np
from utils.utils_training import CE_Loss, Dice_loss

def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, Epoch, epoch_step, epoch_step_val,
                  train_set, val_set, cuda , num_classes, save_dir):
    train_loss = 0
    val_loss = 0

    cls_weights = np.ones([num_classes], np.float32)
    pbar = tqdm(total=epoch_step,desc=f'Epoch(train) {epoch + 1}/{Epoch}',postfix=dict)
    model_train.train()
    for iteration, batch in enumerate(train_set):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda()
                pngs = pngs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()
        #   清零梯度
        optimizer.zero_grad()
        #   前向传播
        outputs = model_train(imgs)
        #   计算损失
        loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
        loss_dice = Dice_loss(outputs, labels)
        loss = loss + loss_dice

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        pbar.set_postfix(**{'train_loss': train_loss / (iteration + 1)})
        pbar.update(1)
    pbar.close()

    pbar = tqdm(total=epoch_step_val, desc=f'Epoch(valid) {epoch + 1}/{Epoch}',postfix=dict)
    model_train.eval()
    for iteration, batch in enumerate(val_set):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda()
                pngs = pngs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()
            #   前向传播
            outputs = model_train(imgs)
            #   计算损失
            loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
            loss_dice = Dice_loss(outputs, labels)
            loss  = loss + loss_dice

            val_loss += loss.item()

            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    pbar.close()

    loss_history.append_loss(epoch + 1, train_loss / epoch_step, val_loss / epoch_step_val)
    #   保存权值
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))