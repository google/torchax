import torch
from torch import nn

import torchax
from torchax import interop
from . import base_test_util


class TiedWeightsModel(nn.Module):
  def __init__(self, vocab_size=10, dim=4):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, dim * 2),
      nn.ReLU(),
      nn.Linear(dim * 2, dim)
    )
    self.head = nn.Linear(dim, vocab_size, bias=False)

    # Tie the weights: head.weight should be the same parameter as emb.weight
    self.head.weight = self.emb.weight

  def forward(self, x):
    x = self.emb(x)
    x = self.mlp(x)
    x = self.head(x)
    return x


class TiedWeightsTest(base_test_util.TestCase):
  def test_tied_weights(self):
    env = torchax.default_env()
    model = TiedWeightsModel()
    self.assertTrue(model.head.weight is model.emb.weight)

    x = torch.randint(0, 10, (2, 5))
    res_torch = model(x)
    
    jax_weights, jax_func = torchax.extract_jax(model, tie_weights=True)
    
    # Verify that the JAX weights are tied (deduplicated)
    self.assertIn("emb.weight", jax_weights)
    self.assertNotIn("head.weight", jax_weights)

    arg = env.t2j_copy(x)
    res_jax = jax_func(jax_weights, (arg,))
    res_jax_torch = env.j2t_copy(res_jax)
    
    self.assertTrue(torch.allclose(res_torch, res_jax_torch, atol=1e-5))

if __name__ == "__main__":
  base_test_util.main()
