# %%
import torch
import torch.nn as nn

import autoencoder.contractive as cae
from dataloaders import get_loaders

# %%
batch_size = 64
train_loader, valid_loader, test_loader = get_loaders(batch_size=batch_size)


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
model.summary()
# model.store_summary()
model_history = model.fit(
    train_loader, val_loader=valid_loader, epochs=epochs, lr=0.002
)
# model.plot_training_history(store=True)
# model.load_checkpoint()
# model.present_latent(valid_loader, store=True)

# torch.save(model, "entire_model.pth")

# model.visualize_reconstructions(test_loader)
