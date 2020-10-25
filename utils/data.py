import librosa
import librosa.display
import glob
from torch.utils.data import Dataset

class StutterData(Dataset):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = []
        
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
            label = 0
        else: 
            label = 1
        sample ={'mfcc': mfcc, 'label':label}
        
        return sample
    
    def visualize(self,idx):
        librosa.display.specshow(sd[idx]['mfcc'], x_axis='time')
        
def 