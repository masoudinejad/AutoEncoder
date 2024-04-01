from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def get_loaders(batch_size=64) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data_folder = "./data/patch_new/patch_train_no_bg/nuclei_patches/reorganized"
    valid_data_folder = "./data/patch_new/patch_valid_no_bg/nuclei_patches/reorganized"
    test_data_folder = "./data/patch_new/test_no_bg"
    train_dataset = datasets.ImageFolder(root=train_data_folder, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_data_folder, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_data_folder, transform=transform)

    # Create DataLoader instances for train and validation sets
    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader
