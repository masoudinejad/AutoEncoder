import torch

from maker.cnn import CNN_model, get_dimensions

model_specs = {
    "type": "CNN",
    "enc_num_channels": [2, 3, 5],
    "enc_kernels": [3, 4, 5],
    "enc_activations": ["relu"],
    "output_activation": "sigmoid",
}

input_shape = torch.Size([3, 22, 28])
dummy_input = torch.randn(1, 3, 22, 28)

model = CNN_model(input_shape, model_specs, latent_dim=2)
encoder, decoder = model.make()
# print(encoder)
print(f"decoder:\n{decoder}")

# latent = get_dimensions(model.enc_layers, input_shape)
# print(latent)
# dummy_dec_input = torch.randn(1, 2)
# out = decoder(dummy_dec_input)
out = decoder(encoder(dummy_input))
print(out.size())
