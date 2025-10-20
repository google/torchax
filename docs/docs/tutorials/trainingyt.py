# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="H2qVWNg4syO_"
#
#
# Training a PyTorch model with JAX
# =====================
#

# %% [markdown] id="fb039419"
#
#
# Introduction
# ------------
#
# This tutorial notebook is adapted from https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
#
# It will keep the most PyTorch code unchanged (especially the model definition),
# and will replace the standard PyTorch train loop (`loss.backward()` + `optimizer.step()` pattern)
# with a JAX train loop (`jax.grad` followed by `optax.apply_updates`).
#
# The rest of the tutorial, such as data loading, print loss etc. are kept as close to the original as possible.
#
# Dataset and DataLoader
# ----------------------
#
# The `Dataset` and `DataLoader` classes encapsulate the process of
# pulling your data from storage and exposing it to your training loop in
# batches.
#
# The `Dataset` is responsible for accessing and processing single
# instances of data.
#
# The `DataLoader` pulls instances of data from the `Dataset` (either
# automatically or with a sampler that you define), collects them in
# batches, and returns them for consumption by your training loop. The
# `DataLoader` works with all kinds of datasets, regardless of the type of
# data they contain.
#
# For this tutorial, we'll be using the Fashion-MNIST dataset provided by
# TorchVision. We use `torchvision.transforms.Normalize()` to zero-center
# and normalize the distribution of the image tile content, and download
# both training and validation data splits.
#

# %%
# Optional: install dependencies
# !pip install matplotlib torch torchax jax optax tensorboard

# %% id="GyJEP81WsyO9"
# For tips on running notebooks in Google Colab, see
# https://docs.pytorch.org/tutorials/beginner/colab
# %matplotlib inline

# %% colab={"base_uri": "https://localhost:8080/"} id="48h6g79csyPB" outputId="0563dbdc-b9d9-4636-a13c-5aee818f879a"
import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

# %% [markdown] id="3uV8TxRTsyPC"
# As always, let's visualize the data as a sanity check:
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 211} id="FObQHGljsyPC" outputId="46d0be39-7817-4dd2-bb0a-da8c995cbcd8"
import matplotlib.pyplot as plt
import numpy as np

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print('  '.join(classes[labels[j]] for j in range(4)))

# %% [markdown] id="bwJeNNVpsyPC"
# The Model
# =========
#
# The model we'll use in this example is a variant of LeNet-5 - it should
# be familiar if you've watched the previous videos in this series.
#

# %% id="MXyOntrgsyPC"
import torch.nn as nn
import torch.nn.functional as F

# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = GarmentClassifier()

# %%
model(images)

# %% [markdown] id="YUzquGuBsyPD"
# Loss Function
# =============
#
# For this example, we'll be using a cross-entropy loss. For demonstration
# purposes, we'll create batches of dummy output and label values, run
# them through the loss function, and examine the result.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="5j5HHPvrsyPD" outputId="b5a6be6c-e0b6-4206-cc60-27a5449cf0ed"
loss_fn = torch.nn.CrossEntropyLoss()

# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_outputs = torch.rand(4, 10)
# Represents the correct class among the 10 being tested
dummy_labels = torch.tensor([1, 5, 3, 7])

print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print('Total loss for this batch: {}'.format(loss.item()))

# %% [markdown]
# ## Move model to 'jax' device

# %%
import torchax
torchax.enable_globally()
model.to('jax')
images = images.to('jax')
dummy_labels = dummy_labels.to('jax')

# %% [markdown] id="HiOna1CzsyPD"
# Optimizer
# =========
#
# For this example, we'll be using simple [optax](https://optax.readthedocs.io/en/latest/getting_started.html)
# optimizer.
#
#

# %% id="tzcmT1YIsyPD"
import optax
start_learning_rate = 1e-3
optimizer = optax.adam(start_learning_rate) 

# %%
print(optimizer)


# %% [markdown] id="7ubeUOe6syPD"
# The Training Loop
# =================
#
# Below, we have a function that performs one training epoch.
#
# First, let's articulate what the training step does.
#
# At each training step, we first evaluate the model. the Model is a
# function that maps the `(weights, input data)` to `prediction`.
#
# $$ model: (weights, input) \mapsto pred $$
#
# In PyTorch, we can use [torch.func.functional_call](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html) to call a model 
# with weights passed in as a paramter.
#
# The loss is a function that takes the prediction, the label to a real number
# representing the loss:
#
# $$ loss: (pred, label) \mapsto loss $$
#
# To train the model, we a glorified Gradient Descent (in this case Adam), so
# we need to have another function that represent the gradient of the 
# loss with respect of weights.
#
# $$ \frac {d loss} {d weights}$$
#
# Finally, the `train_step` itself is a function that takes (weights, optimizer_state, input_data) to
# (updated weights, and updated optimizer_states).
#
# We can spell out the individual components of a train loop, and use Python to assemble them together:

# %%
weights = model.state_dict()

def run_model_and_loss(weights, inputs, labels):
    # First call the model with passed in weights
    output = torch.func.functional_call(model, weights, args=(inputs, ))
    loss = loss_fn(output, labels)
    return loss


# %%
run_model_and_loss(model.state_dict(), images, dummy_labels)

# %% [markdown]
# Now let's define the gradient function of it. In JAX, one would use `jax.jit`. 
# However, `jax.jit` need to take a JAX function (function that takes jax.Array as inputs and outputs) as 
# argument, and here `run_model_and_loss` takes torch.Tensor as inputs / outputs.
#
# One way to solve this issue is to use `jax_view` from the [torchax.interop module](https://github.com/google/torchax/blob/main/torchax/interop.py)
#
# `jax_view` converts a torch function to a jax function.
#
# `torchax` has common JAX functions wrapped in the [so they work with torch-functions as well.
# in this case, we will use `jax_value_and_grad`.

# %%
from torchax.interop import jax_view
import jax

grad_fn_jax = jax.grad( jax_view(run_model_and_loss))

grad_fn_jax(jax_view(weights), jax_view(images), jax_view(dummy_labels)).keys()

# %% [markdown]
# Note that above `grad_fn_jax` is the gradient of `jax_view(run_model_and_loss)` and is a jax function.
#
# if instead we wish to make a it into a torch function, we can use `torch_view` on it and it will
# become a function that takes torch tensors and returns torch tensors.
#
# In fact, the pattern of calling, `jax_view` + `jax.value_and_grad` + `torch_view` is common enough that
# we provided this very wraper as `torchax.interop.jax_value_and_grad` below

# %%
grad_fn = torchax.interop.jax_value_and_grad(run_model_and_loss)




# %% [markdown]
# Now let's assemble the train loop:

# %% id="ei-0yCbisyPD"
# Initialize optimizer
from torchax.interop import call_jax

# Initialize optimizer, we need to call optimizer.init, but
# it is a JAX-function (function that takes jax arrays as input),
# so we use call_jax to pass it torch values:

opt_state = call_jax(optimizer.init, weights)


def train_one_epoch(epoch_index, tb_writer):
    global weights
    global opt_state
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to('jax')
        labels = labels.to('jax')

        # compute gradients
        loss, gradients = grad_fn(weights, inputs, labels)
        # compute updates
        updates, opt_state = call_jax(optimizer.update, gradients, opt_state)
        #apply updates
        weights = call_jax(optax.apply_updates, weights, updates)
        
        # Gather data and report
        running_loss += loss.cpu().item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
        if i > 2000: 
            break
            # NOTE: make it run faster for CI

    return last_loss


# %% [markdown]
# The above will work, however, the grad / optimizer update / apply update is pretty 
# standard; so we have a helper to do exactly that [make_train_step](https://github.com/google/torchax/blob/f41e3de8526f9d4e8410bfb84660faaaf0b3ba4a/torchax/train.py#L28)
#
# Now let's use that instead.
#
# Having a variable for the function of one training step also allows us to compile it with `jax.jit`.
# Here we use `interop.jax_jit` which just wraps `jax.jit` with `torch_view` and pass kwargs verbatim to the 
# underlying `jax.jit` as below.
#
# We can optionally donate the weight and optmizer state, so XLA can issue in-place updates for those 2.
#
#

# %%
import functools
from torchax.train import make_train_step


# the calling convention to make_train_step is the model_fn
# takes weights (trainable params) and buffers (non-trainable params)
# separately. because jax.jit will compute gradients wrt the first arg.
def model_fn(weights, buffers, data):
    return torch.func.functional_call(model, (weights, buffers), data)


one_step = make_train_step(
    model_fn=model_fn,
    loss_fn=loss_fn,
    optax_optimizer=optimizer)


# def one_step(weights, opt_state, inputs, labels):
#             # compute gradients
#     loss, gradients = grad_fn(weights, inputs, labels)
#         # compute updates
#     updates, opt_state = call_jax(optimizer.update, gradients, opt_state)
#         #apply updates
#     weights = call_jax(optax.apply_updates, weights, updates)
#     return loss, weights, opt_state

one_step = torchax.interop.jax_jit(one_step, kwargs_for_jax_jit={'donate_argnums': (0, 2)})


def train_one_epoch(epoch_index, tb_writer):
    global weights
    global opt_state
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to('jax')
        labels = labels.to('jax')

        loss, weights, opt_state = one_step(weights, {}, opt_state, inputs, labels) 
        # Gather data and report
        running_loss += loss.cpu().item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
        if i > 2000: 
            break
            # NOTE: make it run faster for CI

    return last_loss


# %% [markdown] id="mTrYAn1LsyPE"
# Per-Epoch Activity
# ==================
#
# There are a couple of things we'll want to do once per epoch:
#
# -   Perform validation by checking our relative loss on a set of data
#     that was not used for training, and report this
# -   Save a copy of the model
#
# Here, we'll do our reporting in TensorBoard. This will require going to
# the command line to start TensorBoard, and opening it in another browser
# tab.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="dfcNC9UwsyPE" outputId="ea17f7c0-a586-4519-acb5-d49e2923c14c"
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 2

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to('jax')
            vlabels = vlabels.to('jax')
            model.load_state_dict(weights) # put the trained weight back to test it
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

            if i > 1000:
                break

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()


    epoch_number += 1

# %% [markdown]
# ## Save the model checkpoint
#
# Currently `torch.save` (which is based on Pickle) are not able to save tensors on 'jax' device. 
# Because JAX arrays cannot be pickled.
#
# So now we have 2 strategies for saving:
# 1. convert the tensors on jax devices to plain JAX arrays; then use flax.checkpoint to save the data. You will get an JAX-style checkpoint (directory) if you do so.
# 2. convert the tensors from jax devices to CPU torch.Tensor, then use `torch.save`; you will get a regular pickle based checkpoint if you do so.
#
# We recommend 1. and we have provided wrapper in `torchax.save_checkpoint` that does exactly this.

# %%
import os
import orbax.checkpoint as ocp
ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')
model_path = ckpt_dir / 'state'
torchax.save_checkpoint(weights, model_path, step=1)


# %%
# !find /tmp/my-checkpoints/


# %% [markdown]
# You can also produce a torch pickle based checkpoint by moving the state_dict to CPU
#
# You can do so with 

# %%

cpu_state_dict = jax.tree.map(lambda a: a.jax(), weights)

# %%
torch.save(cpu_state_dict, ckpt_dir / 'torch_checkpoint.pkl')

# %%
# !ls /tmp/my-checkpoints/


# %%

# %%

# %%

# %%
