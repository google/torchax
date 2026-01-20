# Copyright 2026 Google LLC
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

"""An example of PEFT LoRA training with torchax.

This script demonstrates how to use torchax for Parameter-Efficient Fine-Tuning
(PEFT) of a language model using Low-Rank Adaptation (LoRA). It downloads a
model and dataset from the Hugging Face Hub for fine-tuning.

To run this example, you need to install the following dependencies:
pip install torchax peft transformers datasets optax

You may also need to be logged in to your Hugging Face account:
`huggingface-cli login`

Need folowing dependencies:
Dependencies include: torchax (library), transformers/datasets (model/data), peft (LoRA), optax (optimizer), and flax (backend).

To run the script:
python examples/peft_lora_training.py
"""

import getpass
import os
import tempfile
from typing import Any, Dict, Tuple

# Fix for environments where a temporary directory is not found
try:
  tempfile.gettempdir()
except FileNotFoundError:
  # If no temp dir is found, create one in the current working directory.
  # This is a common issue in certain restricted environments.
  tmp_dir = os.path.join(os.getcwd(), ".tmp")
  os.makedirs(tmp_dir, exist_ok=True)
  os.environ["TMPDIR"] = tmp_dir
  tempfile.tempdir = tmp_dir

import datasets
import numpy as np
import optax
import peft
import torch
import transformers
from torch.utils.data import DataLoader

import torchax as tx
import torchax.train


def load_datasets(
  model_path: str, dataset_path: str
) -> Tuple[transformers.PreTrainedTokenizer, DataLoader]:
  """Loads tokenizer and training dataloader."""
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  raw_dataset = datasets.load_dataset(dataset_path)["train"]

  def format_example(example):
    chat = [
      {"role": "user", "content": example["instruction"]},
      {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False)
    inputs = tokenizer(
      text,
      padding="max_length",
      max_length=256,
      truncation=True,
      return_tensors=None,
    )
    return inputs

  dataset = raw_dataset.map(
    format_example,
    load_from_cache_file=False,
    cache_file_name=os.path.join(tempfile.gettempdir(), "cache.arrow"),
    remove_columns=["instruction", "input", "output", "text"],
  )

  collator = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
  )

  train_dataloader = DataLoader(
    dataset,
    shuffle=False,
    collate_fn=collator,
    batch_size=2,
  )
  return tokenizer, train_dataloader


def create_peft_model(model_path: str) -> torch.nn.Module:
  """Loads base model and applies PEFT configuration."""
  peft_config = peft.LoraConfig(
    task_type=peft.TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
  )
  model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
  )
  model = peft.get_peft_model(model, peft_config)
  model.print_trainable_parameters()
  return model


def train_model(
  model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> Dict[str, Any]:
  """Runs the training loop and returns trained parameters."""
  print("Starting training...")
  with tx.default_env():
    model = model.to(device)
    model.train()

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())
    frozen_params = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    buffers.update(frozen_params)

    optimizer = optax.adam(1e-4)
    opt_state = tx.interop.call_jax(optimizer.init, params)

    def model_fn(weights, buffers, batch):
      return torch.func.functional_call(
        model, {**weights, **buffers}, args=(), kwargs=batch
      )

    def loss_fn(model_output, labels):
      return model_output.loss

    step_fn = tx.train.make_train_step(model_fn, loss_fn, optimizer)

    for step, batch in enumerate(dataloader):
      batch = {k: v.to("jax") for k, v in batch.items()}
      loss, params, opt_state = step_fn(
        params, buffers, opt_state, batch, batch["labels"]
      )
      print(f"Step {step}, Loss: {loss.item()}")

      if step >= 5:
        break
  print("Training finished.")
  return params


def save_model_and_tokenizer(
  model: torch.nn.Module,
  tokenizer: transformers.PreTrainedTokenizer,
  params: Dict[str, Any],
  save_path: str,
) -> None:
  """Saves the trained PEFT model and tokenizer."""
  print(f"Saving PEFT model to {save_path}")
  with torch.no_grad():
    # 1. Create a temporary CPU state_dict
    cpu_lora_state_dict = {
      name: torch.tensor(np.array(p)).contiguous() for name, p in params.items()
    }

    # 2. Save directly using the state_dict argument
    model.save_pretrained(save_path, state_dict=cpu_lora_state_dict)
  tokenizer.save_pretrained(save_path)
  print(f"Model saved to {save_path}")


def run_inference(
  model_path: str,
  peft_path: str,
  dataloader: DataLoader,
  device: torch.device,
) -> None:
  """Loads the saved model and runs inference on one batch."""
  print("\n--- Starting Inference ---")
  with tx.default_env(), torch.no_grad():
    # Get batch
    batch = {}
    for b in dataloader:
      batch = b
      break
    batch = {k: v.to(device) for k, v in batch.items()}

    # Load models with torchax disabled to avoid dispatch errors
    with tx.disable_temporarily():
      print("Loading base model for inference...")
      base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
      )
      print("Loading PEFT model for inference...")
      peft_base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
      )
      peft_model = peft.PeftModel.from_pretrained(peft_base_model, peft_path)

    # Base model inference
    base_model.to(device).eval()
    print("Running inference on base model...")
    base_outputs = base_model(**batch)
    print(f"Base model loss: {base_outputs.loss.item()}")
    del base_model  # Free memory

    # PEFT model inference
    peft_model.to(device).eval()
    print("Running inference on PEFT-loaded model...")
    peft_outputs = peft_model(**batch)
    print(f"PEFT-loaded model loss: {peft_outputs.loss.item()}")


def main():
  torch.manual_seed(0)
  tx.enable_accuracy_mode()
  device = torch.device("jax")
  print(f"Using device: {device}")

  # This example downloads a model and dataset from the Hugging Face Hub.
  model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  dataset_path = "tatsu-lab/alpaca"
  peft_save_path = f"/tmp/txla2_peft_lora_example_model_{getpass.getuser()}"

  tokenizer, train_dataloader = load_datasets(model_path, dataset_path)
  model = create_peft_model(model_path)

  params = train_model(model, train_dataloader, device)

  save_model_and_tokenizer(model, tokenizer, params, peft_save_path)

  run_inference(model_path, peft_save_path, train_dataloader, device)


if __name__ == "__main__":
  main()
