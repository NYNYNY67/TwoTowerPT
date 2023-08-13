import torch
from torch.utils.data import Dataset, DataLoader


def get_user_item_loader(
        users,
        items,
        batch_size=128,
        shuffle=True,
):
    dataset = UserItemDataset(users, items)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class UserItemDataset(Dataset):
    def __init__(
            self,
            users,
            items,
    ):
        super().__init__()
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]


def get_user_loader(
        users,
        batch_size=128,
        shuffle=False,
):
    dataset = UserDataset(users)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class UserDataset(Dataset):
    def __init__(
            self,
            users,
    ):
        super().__init__()
        self.users = users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx]