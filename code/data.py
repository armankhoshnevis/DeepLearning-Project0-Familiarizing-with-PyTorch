import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, alpha, beta, num_samples, split, seed):
        """Custom dataset class"""
        super().__init__()
        
        if split == 'train':
            torch.manual_seed(seed)
        elif split == 'val':
            torch.manual_seed(seed + 1)
        elif split == 'test':
            torch.manual_seed(seed + 2)
        else:
            raise ValueError('Invalid split')

        self.x = torch.randn(num_samples, device='cpu')
        e = torch.randn(num_samples, device='cpu')
        self.y = alpha * self.x + beta + e
    
    def __len__(self):
        """Returns the total number of available samples"""
        return len(self.x)

    def __getitem__(self, index):
        """Returns the x and y values at the given index"""
        return self.x[index].view(1), self.y[index].view(1)