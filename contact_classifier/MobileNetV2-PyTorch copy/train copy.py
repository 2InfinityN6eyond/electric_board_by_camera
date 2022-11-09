from __future__ import print_function, division

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse

from MobileNetV2 import mobilenetv2_19
from contact_dataset import makeTrainValidDataLoaders, makeTestDataLoaders


def train_model(
    args, model,
    train_data_loader, valid_data_loader,
    criterion, optimizer,
    scheduler,
) :
    resumed = False

    for epoch in range(
        args.start_epoch + 1,
        args.start_epoch + 1 + args.num_epochs,
    ) :
        print(
            "{:3d}->{:3d}->{:3d}".format(
                args.start_epoch + 1,
                epoch,
                args.start_epoch + 1 + args.num_epochs,
            ),
        )
       
        # train
        model.train(True)

        accum_data_len = 0
        running_loss = 0.0
        running_corrects = 0
        with tqdm.tqdm(total = len(train_data_loader.dataset), desc="train", unit="img") as pbar :    
            
            for i, (inputs, labels) in enumerate(train_data_loader) :
                if use_gpu :
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else :
                    inputs, labels = Variable(inputs), Variable(labels)
                
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                

                running_loss += loss.item()
                running_corrects = len(list(filter(
                    lambda x : x.sum() == len(x),
                    torch.round(outputs) == labels
                )))

                accum_data_len += inputs.shape[0]
                batch_loss = running_loss / accum_data_len
                batch_acc = running_corrects / accum_data_len

                pbar.update(inputs.shape[0])
                pbar.set_postfix_str(
                    "loss:{:.6f} acc:{:.6f}".format(
                        batch_loss,
                        batch_acc,
                    )
                )

        if args.start_epoch > 0 and (not resumed) :
            scheduler.step(args.start_epoch+1)
            resumed = True
        else :
            scheduler.step(epoch)

        # valid
        model.train(False)
        running_loss = 0.0
        running_corrects = 0
        accum_data_len = 0
        with tqdm.tqdm(total = len(valid_data_loader.dataset), desc="valid", unit="img") as pbar :    
            for i, (inputs, labels) in enumerate(valid_data_loader) :
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                running_corrects = len(list(filter(
                    lambda x : sum(x) == len(x),
                    torch.round(outputs) == labels
                )))
                accum_data_len += inputs.shape[0]
                batch_loss = running_loss / accum_data_len
                batch_acc = running_corrects / accum_data_len

                pbar.update(inputs.shape[0])
                pbar.set_postfix_str(
                    "loss:{:.6f} acc:{:.6f}".format(
                        batch_loss,
                        batch_acc,
                    )
                )

        if (epoch+1) % args.save_epoch_freq == 0 :
            if not os.path.exists(args.save_path) :
                os.makedirs(args.save_path)
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_path,
                    f"epoch_{str(epoch)}_{str(int(time.time()))}.pth",
                )
            )

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data_root_path', type=str, default="../data")
    parser.add_argument('--batch_size',  type=int, default=80)
    parser.add_argument('--val_ratio',   type=float, default=0.2)
    parser.add_argument('--num_class',   type=int, default=6)
    parser.add_argument('--num_epochs',  type=int, default=100)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpus',        type=str, default=0)
    parser.add_argument('--print_freq',  type=int, default=10)
    parser.add_argument('--save_epoch_freq', type=int, default=1)
    parser.add_argument('--save_path',   type=str, default = "./checkpoints")
    parser.add_argument('--resume',      type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    args = parser.parse_args()

    train_data_loader, valid_data_loader = makeTrainValidDataLoaders(
        args.data_root_path,
        args.val_ratio,
        args.batch_size,
    )

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    model = mobilenetv2_19(num_classes = args.num_class)

    if args.resume :
        """
        """
        if os.path.isfile(args.resume) :
            model.load_state_dict(
                torch.load(args.resume)
            )
            args.start_epoch = int(args.resume.split("_")[1]) + 1
            print(f"model {args.resume} loaded, epoch start at {args.start_epoch}")
 
    if use_gpu:
        model = model.cuda()

    # define loss function
    criterion = nn.MultiLabelSoftMarginLoss()

    #optimizer_ft = optim.SGD(
    #    model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004
    #)
    
    optimizer_ft = optim.Adam(
        model.parameters(), lr = args.lr,
    )

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size = 1, gamma = 0.98
    )

    train_model(
        args=args,
        model=model,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
    )
