
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import os
import argparse

from contact_dataset import makeTrainValidDataLoaders, makeTestDataLoaders
from blazehand_contact import BlazeHandContact

torch.set_grad_enabled(True)


def train_model(
    args, model,
    train_data_loader, valid_data_loader,
    criterion, optimizer,
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
    parser.add_argument('--data_root_path', type=str, default="../all")
    parser.add_argument('--batch_size',  type=int, default=140)
    parser.add_argument('--val_ratio',   type=float, default=0.2)
    parser.add_argument('--num_class',   type=int, default=6)
    parser.add_argument('--num_epochs',  type=int, default=100)
    parser.add_argument('--lr',          type=float, default=0.0001)
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

    model = BlazeHandContact()

    if args.resume :
        if os.path.isfile(args.resume) :
            model.load_weights(args.resume)
            if "blaze" in args.resume :
                args.statr_epoch = 0
            else :
                args.start_epoch = int(args.resume.split("_")[1]) + 1
                
            print(f"model {args.resume} loaded, epoch start at {args.start_epoch}")
 
    if use_gpu:
        model = model.cuda()

    criterion = nn.MSELoss()
    optimizer_ft = optim.Adam(
        model.parameters(), lr = args.lr,
    )

    train_model(
        args=args,
        model=model,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        criterion=criterion,
        optimizer=optimizer_ft,
    )
