# syntax=docker/dockerfile:experimental
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use Python 3.10 as the base image
FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk git

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1
RUN pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install optax fire tensorflow tensorboard-plugin-profile
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /
RUN git clone https://github.com/pytorch/torchtitan.git
WORKDIR /torchtitan
RUN pip install -r requirements.txt
RUN pip install .

WORKDIR /
RUN git clone https://github.com/pytorch/xla.git
WORKDIR xla/experimental/torchax
RUN pip install -e .

ENTRYPOINT ["python", "examples/train_llama_torchtitan/train_llama.py"]
CMD ["--batch_size=8", "--seqlen=2048"]