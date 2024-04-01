# %%
# Load packages
import autoencoder.contractive as cae
from dataloaders import get_loaders
from maker.mlp import MLP_model

# %%
# Make data loaders
batch_size = 64
train_loader, valid_loader, test_loader = get_loaders(batch_size=batch_size)
# Find the input shape
inputs, targets = next(iter(train_loader))
input_shape = inputs.shape[1:]  # Remove batch size

# %%
# Define and make model parts
enc_dims = [128, 64, 32]
latent_dimension = 5
model_maker = MLP_model(
    input_shape,
    enc_dims,
    latent_dim=latent_dimension,
)
encoder, decoder = model_maker.make()

# %%
lambda_c = 0.5
name_extras = f"lc{lambda_c:.2f}_b{batch_size}"
# Make the model
model = cae.CAE(
    input_shape=input_shape,
    encoder=encoder,
    decoder=decoder,
    latent_dim=latent_dimension,
    lambda_c=lambda_c,
    name_extras=name_extras,
)

# %%
# Train the model
epochs = 1
print(model)
# model.summary()
model.store_summary()
# model_history = model.fit(
#     train_loader, val_loader=valid_loader, epochs=epochs, lr=0.002
# )
# model.plot_training_history(store=True)

# # %%
# # Evaluate the model
# # load the best performing epoch
# model.load_checkpoint()
# model.present_latent(valid_loader, store=True)
# # Store the best model
# model.store()
# # Visualize the reconstruction of test data
# model.visualize_reconstructions(test_loader)
