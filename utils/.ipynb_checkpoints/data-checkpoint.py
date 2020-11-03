import librosa
import librosa.display
import glob
from torch.utils.data import Dataset
import numpy as np
import torch 
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

class StutterData(Dataset):
    
    def __init__(self, root_dir, single_file=False):
        self.root_dir = root_dir
        self.files = []
        
        if single_file:
            self.files.append(root_dir)
        else:
            for path in glob.glob(root_dir):
                self.files.append(path)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio, sr = librosa.core.load(self.files[idx])
        mfcc = librosa.feature.mfcc(audio, sr)
        
        if 'non-stuttered' in self.files[idx]:
            label = 0.0
        else: 
            label = 1.0
        sample ={'mfcc': mfcc, 'label':label}
        
        return sample
    
    def visualize(self,idx):
        librosa.display.specshow(sd[idx]['mfcc'], x_axis='time')
        
        
def load_data(dataset, batch_size, validation_split=0.2, shuffle_dataset=True, random_seed=42):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:dataset_size-split], indices[dataset_size-split:]
    
    if shuffle_dataset:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
    else:
        train_sampler = SequentialSampler(train_indices)
        valid_sampler = SequentialSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader