import unittest

import torch

import torchax


class OutVariantTest(unittest.TestCase):
  def setUp(self):
    self.env = torchax.default_env()
    self.env.config.debug_print_each_op = True

  def test_add_out(self):
    with self.env:
      a = torch.ones((2, 2), device="jax")
      b = torch.ones((2, 2), device="jax")
      c = torch.ones((2, 2), device="jax")

      torch.add(a, b, out=c)
      torch.testing.assert_close(c, a + b, check_device=False)

  def test_sub_out(self):
    with self.env:
      a = torch.ones((2, 2), device="jax")
      b = torch.ones((2, 2), device="jax")
      c = torch.ones((2, 2), device="jax")

      torch.sub(a, b, out=c)
      torch.testing.assert_close(c, a - b, check_device=False)

  def test_mul_out(self):
    with self.env:
      a = torch.ones((2, 2), device="jax")
      b = torch.ones((2, 2), device="jax")
      c = torch.ones((2, 2), device="jax")

      torch.mul(a, b, out=c)
      torch.testing.assert_close(c, a * b, check_device=False)

  def test_div_out(self):
    with self.env:
      a = torch.ones((2, 2), device="jax")
      b = torch.ones((2, 2), device="jax")
      c = torch.ones((2, 2), device="jax")

      torch.div(a, b, out=c)
      torch.testing.assert_close(c, a / b, check_device=False)
