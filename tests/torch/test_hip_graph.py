"""ROCm/HIP graph capture smoke test (skipped on CPU-only test hosts)."""

import unittest

import torch

from hip_quant.torch_api import capture_hip_graph


@unittest.skipUnless(torch.cuda.is_available(), "requires a ROCm/HIP GPU")
class TestHipGraphCapture(unittest.TestCase):
    def test_replay_matches_eager_and_does_not_alias_default_output(self):
        x = torch.randn(8, 16, device="cuda")
        runner = capture_hip_graph(lambda value: value * 1.5 + 2.0, x)

        first = runner.replay(x)
        second_input = torch.randn_like(x)
        second = runner.replay(second_input)

        torch.testing.assert_close(first, x * 1.5 + 2.0)
        torch.testing.assert_close(second, second_input * 1.5 + 2.0)

    def test_replay_rejects_changed_metadata(self):
        x = torch.randn(8, 16, device="cuda")
        runner = capture_hip_graph(lambda value: value + 1.0, x)
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            runner.replay(torch.randn(4, 16, device="cuda"))
