import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as data

from torchvision.models import resnet18

import lightning as L

from datasets import WRsmallepoch_CWT
from models import ResNet18


def main(data_file, annotation_file, sample_rate=5000):
    
    dataset = WRsmallepoch_CWT(
        data_file = data_file, 
        annotation_file = annotation_file, 
        sample_rate= sample_rate,
        epoch_size=1
    )
    
    
    trv_set_size = int(len(dataset) * 0.8)
    #test_set_size = len(dataset) - trv_set_size

    trv_indices = list(range(trv_set_size))
    #test_indices = list(range(trv_set_size, len(dataset)))


    trv_set = data.Subset(dataset, trv_indices)
    #test_set = data.Subset(dataset, test_indices)
    
    # use 20% of training data for validation
    train_set_size = int(len(trv_set) * 0.8)
    valid_set_size = len(trv_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(trv_set, [train_set_size, valid_set_size], generator=seed)

    train_indices = train_set.indices
    train_weights = 1.0 / dataset.frequencies[train_indices]
    datasampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_set), replacement=True)


    train_loader = DataLoader(train_set, batch_size=200,sampler=datasampler)
    valid_loader = DataLoader(valid_set, batch_size=200)
    #test_loader = DataLoader(test_set, batch_size=500)
    

    model = ResNet18(num_classes=2)

    trainer = L.Trainer(max_epochs=200,log_every_n_steps=15, default_root_dir='/app/Data/resnetcwt', accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, valid_loader)
    #trainer.test(GRUc, dataloaders=test_loader)


if __name__ == "__main__":
    main('/app/Data/WR/WR5_Run4.hdf5', '/app/Data/WR/Annotations/1sec_epocs.pkl')

