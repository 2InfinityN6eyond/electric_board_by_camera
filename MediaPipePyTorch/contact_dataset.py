from torchvision import transforms, datasets
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

DATA_ROOT_PATH = "../data"

def makeTrainValidDataLoaders(
    data_root_path,
    val_ratio,
    batch_size
) :

    print(data_root_path)
    whole_dataset = ContactDataSet(
        data_root_path,
        list(filter(
            lambda image_path : "png" in image_path,
            os.listdir(data_root_path)
        ))
    )

    print(len(whole_dataset))

    train_idx, val_idx = train_test_split(
        list(range(len(whole_dataset))), test_size = val_ratio
    )
    train_dataset = Subset(whole_dataset, train_idx)
    valid_dataset = Subset(whole_dataset, val_idx)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=True
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size = batch_size,
        shuffle=True
    )
    return train_data_loader, valid_data_loader 


def makeTestDataLoaders(
    data_root_path,
    batch_size
) :
    test_dataset = ContactDataSet(
        data_root_path,
        list(filter(
            lambda image_path : "png" in image_path,
            os.listdir(data_root_path)
        )),
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle=True
    )
    return test_data_loader 

class ContactDataSet(Dataset) :
    def __init__(
        self,
        data_root_path,
        image_paths,
    ) :
        super(ContactDataSet, self).__init__()
        self.data_root_path = data_root_path
        self.image_paths = image_paths

    def __len__(self) :
        return len(self.image_paths)

    def __getitem__(self, item) :
        image_path = self.image_paths[item]

        image = cv2.imread(self.data_root_path + "/" +image_path )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #.transpose((2, 0, 1))

        label = image_path.split("_")[-1].split(".")[0]

        return (
            torch.Tensor(image.transpose((2, 0, 1)) / 255.0),
            torch.Tensor(list(map(
                lambda chararc : int(chararc),
                label
            )))
        )