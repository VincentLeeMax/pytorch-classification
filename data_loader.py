import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

def get_data_loader(data_dir):
    num_workers = {'train': 10, 'val': 0, 'test': 0}
    dataset_dict = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225]),
                                            ])) for x in ['train', 'val', 'test']}

    dataloader_dict = {x: torch.utils.data.DataLoader(dataset_dict[x], batch_size=100,
                                                      shuffle=True, num_workers=num_workers[x])
                       for x in ['train', 'val', 'test']}

    return dataloader_dict['train'], dataloader_dict['val'], dataloader_dict['test']