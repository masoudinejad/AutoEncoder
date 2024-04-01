import torch

from autoencoder.autoencoder import Autoencoder


class MLP_AE(Autoencoder):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        latent_dim=2,
        reconstruction_loss_function=torch.nn.MSELoss(),
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_function = reconstruction_loss_function
        self.history = {"tr_re": [], "val_re": []}
        self.latent_dim = latent_dim
