import functools
import unittest

import jax
import optax
import torch
import torch.nn as nn

import torchax
from torchax import interop


class EmbeddingTestModule(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, padding_idx):
    super().__init__()
    self.embedding = nn.Embedding(
      num_embeddings=num_embeddings,
      embedding_dim=embedding_dim,
      padding_idx=padding_idx,
    )

  def forward(self, x):
    output_embeddings = self.embedding(x)
    loss = output_embeddings.sum()
    return loss, output_embeddings


class TestEmbeddingPaddingIdx(unittest.TestCase):
  def setUp(self):
    self.env = torchax.default_env()
    torch.manual_seed(0)

  def test_embedding_grad_with_padding_idx_jitted(self):
    """
    Tests that the gradient for the padding_idx in an embedding layer is zero
    after a backward pass in the torchax environment.
    """
    vocab_size = 10
    embedding_dim = 4
    padding_idx = 0
    batch_size = 5
    seq_len = 8

    model = EmbeddingTestModule(vocab_size, embedding_dim, padding_idx=0)
    # Create input data that includes the padding_idx
    input_indices = torch.randint(1, vocab_size, (batch_size, seq_len))
    input_indices[0, :] = padding_idx  # Ensure padding_idx is present

    weights_copy = model.state_dict()["embedding.weight"].clone()

    def run_train(model):
      with self.env:
        # Convert model and inputs to the torchax backend
        tx_model, (tx_input,) = self.env.to_xla((model, (input_indices,)))

        # 4. Perform forward pass, calculate loss, and backpropagate
        # We need to be in train mode for gradients to be computed
        tx_model.train()

        model_jittable = torchax.interop.JittableModule(tx_model)

        optimizer = optax.sgd(
          learning_rate=0.1,
        )
        weights = model_jittable.params
        buffers = model_jittable.buffers

        model_fn = functools.partial(
          model_jittable.functional_call,  # type: ignore
          "forward",
        )

        train_step = make_train_step(
          model_fn,
          optimizer,
        )
        train_step_fn = interop.jax_jit(
          train_step,
          kwargs_for_jax_jit={
            "donate_argnums": (0,),
          },
        )

        opt_state = interop.call_jax(optimizer.init, weights)

        loss, output_embeddings, gradient = None, None, None
        for _i in range(5):
          loss, output_embeddings, gradient, weights, opt_state = train_step_fn(
            weights, buffers, opt_state, tx_input
          )

        return loss, output_embeddings, gradient, weights

    _loss, output_embeddings, gradient, weights = run_train(model)

    model = EmbeddingTestModule(vocab_size, embedding_dim, padding_idx=None)
    with self.env:
      assert torch.all(output_embeddings[0, :] == 0.0)  # type: ignore
      assert torch.all(gradient["embedding.weight"][0, :] == 0.0)  # type: ignore
      assert torch.all(weights["embedding.weight"][0, :] == 0.0)

      # non padded model will still has 0-s at the padding index row but they are now learnable
      model.load_state_dict({"embedding.weight": weights_copy})

    _loss2, output_embeddings2, gradient2, weights2 = run_train(model)
    with self.env:
      assert not torch.all(gradient2["embedding.weight"][0, :] == 0.0)  # type: ignore
      assert not torch.all(weights2["embedding.weight"][0, :] == 0.0)
      # Compare the weights for all non-padded rows.
      torch.testing.assert_close(
        output_embeddings[1:, :],  # type: ignore
        output_embeddings2[1:, :],  # type: ignore
        rtol=1e-6,
        atol=1e-6,
      )
      torch.testing.assert_close(
        weights["embedding.weight"][1:, :],
        weights2["embedding.weight"][1:, :],
        rtol=1e-6,
        atol=1e-6,
      )
      torch.testing.assert_close(
        gradient["embedding.weight"][1:, :],  # type: ignore
        gradient2["embedding.weight"][1:, :],  # type: ignore
        rtol=1e-6,
        atol=1e-6,
      )


def make_train_step(
  model_fn,
  optax_optimizer,
):
  """
  A slightly modified version of torchax.train.make_train_step()
  """
  env = torchax.default_env()

  def loss_and_aux(weights, buffers, *args):  # inputs are XLATensor
    with env, jax.named_scope("compute_loss"):
      model_out = model_fn(weights, buffers, *args)
      return model_out[0], model_out[1]

  grad_fn = interop.jax_value_and_grad(loss_and_aux, {"has_aux": True})

  def step(weights, buffers, opt_state, *args):  # inputs are array
    with jax.named_scope("compute_gradient"):
      (loss, aux_data), gradient = grad_fn(weights, buffers, *args)

    with jax.named_scope("optimizer_updates"):
      updates, opt_state = interop.call_jax(
        optax_optimizer.update, gradient, opt_state, weights
      )  # type: ignore
      weights = interop.call_jax(optax.apply_updates, weights, updates)

    return loss, aux_data, gradient, weights, opt_state

  return step


if __name__ == "__main__":
  unittest.main()
