import copy
import math
from abc import abstractmethod
from collections import OrderedDict
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from train_AE_pt import fit_AE

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Defining a meta-model with all basic functions
class Meta_Autoencoder(torch.nn.Module):
    @abstractmethod
    def __init__(self, latent_dim=2, mode="basic"):
        super().__init__()
        assert latent_dim > 0
        self.mode = mode
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    # def evaluate(self, criterion, data, targets, bs=None, **kwargs):
    #     bs = len(data) if bs is None else bs
    #     n_batches = math.ceil(len(data) / bs)
    #     if "Conv" in self.__class__.__name__:
    #         data = data.to("cpu")
    #         targets = targets.to("cpu")
    #     else:
    #         data = data.to(device)
    #         targets = targets.to(device)

    #     # evaluate
    #     self.to(device)
    #     self.eval()
    #     with torch.no_grad():
    #         val_loss = 0
    #         for batch_idx in range(n_batches):
    #             data_batch = data[batch_idx * bs : batch_idx * bs + bs].to(device)
    #             targets_batch = targets[batch_idx * bs : batch_idx * bs + bs].to(device)
    #             outputs = self(data_batch)
    #             # flatten outputs in case of ConvAE (targets already flat)
    #             loss = criterion(torch.flatten(outputs, 1), targets_batch)
    #             val_loss += loss.item()
    #     return val_loss / n_batches

    def fit(
        self, tr_data, val_data, num_epochs=10, bs=32, lr=0.1, momentum=0.0, **kwargs
    ):
        return fit_AE(
            self,
            tr_data,
            val_data,
            mode=self.mode,
            num_epochs=num_epochs,
            bs=bs,
            lr=lr,
            momentum=momentum,
            **kwargs,
        )


def get_activation_fcn(act_fcn):
    match act_fcn:
        case "relu":
            return nn.ReLU(inplace=True)
        case "sigmoid":
            return nn.Sigmoid()


class MLP_AutoEncoder(Meta_Autoencoder):

    def __init__(
        self,
        input_shape,
        enc_dims: Sequence[int],
        latent_dim: int = 2,
        dec_dims=None,
        use_bias=True,
        enc_act_fcn="relu",
        dec_act_fcn=None,
        output_act_fcn="sigmoid",
    ):
        super().__init__()
        assert len(enc_dims) > 0 and all(d > 0 for d in enc_dims)
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.type = "MLP"
        self.enc_dims = enc_dims
        self.use_bias = use_bias
        if isinstance(enc_act_fcn, str):
            self.enc_act_fcn = [enc_act_fcn] * len(self.enc_dims)
        else:
            self.enc_act_fcn = enc_act_fcn
        if dec_dims is None:
            self.dec_dims = list(reversed(self.enc_dims))
        else:
            self.dec_dims = dec_dims
        if dec_act_fcn is None:
            self.dec_act_fcn = list(reversed(self.enc_act_fcn))
        else:
            self.dec_act_fcn = dec_act_fcn

        self.output_act_fcn = output_act_fcn

        self._find_input_feature_size()
        self._build_encoder()
        self._build_decoder()

    def _find_input_feature_size(self):
        # Exclude the batch size from the input shape
        input_shape_without_batch = self.input_shape[1:]
        # Calculate the total number of elements in the input tensor
        input_size = torch.prod(torch.tensor(input_shape_without_batch))
        self.feature_size = input_size.item()

    def _build_encoder(self):
        input_layer = nn.Flatten()
        #  Make layers
        enc_layers = OrderedDict({"input_layer": input_layer})
        input_dims = [self.feature_size] + self.enc_dims[:-1]
        for idx in range(len(self.enc_dims)):
            enc_layers.update(
                {
                    f"enc_fc{idx}": nn.Linear(
                        input_dims[idx],
                        self.enc_dims[idx],
                        bias=self.use_bias,
                    )
                }
            )
            layer_act_fcn = get_activation_fcn(self.enc_act_fcn[idx])
            enc_layers.update({f"act_enc_fc{idx}": layer_act_fcn})
        # Latent space
        enc_layers.update(
            {
                f"latent_space": nn.Linear(
                    self.enc_dims[-1],
                    self.latent_dim,
                    bias=self.use_bias,
                )
            }
        )

        self.encoder = nn.Sequential(enc_layers)

    def _build_decoder(self):
        input_dims = [self.latent_dim] + self.dec_dims[:-1]
        #  Make layers
        dec_layers = OrderedDict()
        for idx in range(len(self.dec_dims)):
            dec_layers.update(
                {
                    f"dec_fc{idx}": nn.Linear(
                        input_dims[idx],
                        self.dec_dims[idx],
                        bias=self.use_bias,
                    )
                }
            )
            layer_act_fcn = get_activation_fcn(self.dec_act_fcn[idx])
            dec_layers.update({f"act_dec_fc{idx}": layer_act_fcn})
        # Latent space
        dec_layers.update(
            {
                f"output_flat_layer": nn.Linear(
                    self.dec_dims[-1],
                    self.feature_size,
                    bias=self.use_bias,
                )
            }
        )
        layer_act_fcn = get_activation_fcn(self.output_act_fcn)
        dec_layers.update({"output_flat_act_fcn": layer_act_fcn})
        dec_layers.update({"output_layer": nn.Unflatten(1, self.input_shape[1:])})

        self.decoder = nn.Sequential(dec_layers)


# class LSTM_Autoencoder(torch.nn.Module):

#     def __init__(self, seq_len, num_features, embedding_dim=50):
#         super(LSTM_Autoencoder, self).__init__()
#         self.seq_len = seq_len
#         self.num_features = num_features
#         self.embedding_dim = embedding_dim
#         self.encoder = None
#         self.decoder = None

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)

#         return x
