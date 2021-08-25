from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples