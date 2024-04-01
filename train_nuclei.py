# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import autoencoder.contractive as cae

# %%
# Define a transform to normalize the data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.view(1, -1)),
    ]
)
train_data_folder = "./data/cell/patch_new/patch_train_no_bg/nuclei_patches/reorganized"
valid_data_folder = "./data/cell/patch_new/patch_valid_no_bg/nuclei_patches/reorganized"
test_data_folder = "./data/cell/patch_new/test_no_bg"
train_dataset = datasets.ImageFolder(root=train_data_folder, transform=transform)
valid_dataset = datasets.ImageFolder(root=valid_data_folder, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_folder, transform=transform)

# Create DataLoader instances for train and validation sets
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# %%
def _make_encoder():
    # Encoder as a sequential model
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=3 * 79 * 79, out_features=128),
        # nn.Linear(in_features=784, out_features=256),
        # nn.ReLU(),
        # nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=5),
    )
    return encoder


def _make_decoder():
    # Decoder as a sequential model
    decoder = nn.Sequential(
        nn.Linear(in_features=5, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=3 * 79 * 79),
        # nn.Linear(in_features=128, out_features=256),
        # nn.ReLU(),
        # nn.Linear(in_features=256, out_features=784),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Unflatten(1, (3, 79, 79)),
    )
    return decoder


# %%
# Find the input shape
inputs, targets = next(iter(train_loader))
img_shape = inputs.shape[1:]
model = cae.CAE(
    input_shape=img_shape,
    encoder=_make_encoder(),
    decoder=_make_decoder(),
    latent_dim=5,
    lambda_c=2,
)
# %%
epochs = 1
# model.summary()
# model.store_summary()
model_history = model.fit(
    train_loader, val_loader=valid_loader, epochs=epochs, lr=0.002
)
# model.plot_training_history(store=True)
# model.load_checkpoint()
# model.present_latent(valid_loader, store=True)

# torch.save(model, "entire_model.pth")

# %%
model.visualize_reconstructions(test_loader)
