from torchvision import transforms, datasets
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image

DATA_ROOT_PATH = "../data"

def makeTrainValidDataLoaders(
    data_root_path,
    val_ratio,
    batch_size
) :
    whole_dataset = ContactDataSet(
        data_root_path,
        list(filter(
            lambda image_path : "png" in image_path,
            os.listdir(DATA_ROOT_PATH)
        )),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
        ])
    )
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
            os.listdir(DATA_ROOT_PATH)
        )),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
        ])
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
        data_transforms
    ) :
        super(ContactDataSet, self).__init__()
        self.data_root_path = data_root_path
        self.image_paths = image_paths
        self.data_transforms = data_transforms

    def __len__(self) :
        return len(self.image_paths)

    def __getitem__(self, item) :
        image_path = self.image_paths[item]
        image = Image.open(os.path.join(
            self.data_root_path,
            image_path,
        ))
        label = image_path.split("_")[-1].split(".")[0]
        
        return (
            self.data_transforms(image),
            torch.Tensor(list(map(
                lambda chararc : int(chararc),
                label
            )))
        )