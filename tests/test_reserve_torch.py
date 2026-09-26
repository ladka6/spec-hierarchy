"""Real tiny-model correctness tests, without model downloads."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from dflash.model import DFlashDraftModel, extract_context_feature

from hspec.pipeline import ar_generate
from hspec.reserve import ReserveConfig
from hspec.reserve_torch import TorchReserveBackend, reserve_generate


def tiny_models():
    torch.manual_seed(17)
    cfg = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=16, max_position_embeddings=256, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        target.lm_head.weight.mul_(8)
    mid = copy.deepcopy(target)
    with torch.no_grad():
        for p in mid.parameters():
            p.add_(torch.randn_like(p) * 0.2 * p.std())
    cfg = Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=16, max_position_embeddings=256, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    cfg.num_target_layers = 4
    cfg.dflash_config = {"block_size": 4, "mask_token_id": 47}
    draft = DFlashDraftModel(cfg).eval()
    return draft, target, mid


class TorchReserveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_lossless_across_schedules_and_short_limits(self):
        draft, target, mid = tiny_models()
        for prompt in (torch.tensor([[1, 2, 3]]), torch.tensor([[7, 3, 9, 8, 1]])):
            for limit in (1, 2, 17):
                ref = ar_generate(target, prompt, limit, []).output_ids
                for cfg in (ReserveConfig(0, synchronous=True),
                            ReserveConfig(0, target_fallback=False),
                            ReserveConfig(2, target_fallback=False, max_ahead=3),
                            ReserveConfig(2, target_fallback=True)):
                    with self.subTest(prompt=prompt.tolist(), limit=limit, cfg=cfg):
                        r = reserve_generate(draft, target, mid, prompt, limit, [], cfg,
                                             fork_threshold=1.0)
                        self.assertTrue(torch.equal(r.output_ids, ref), (r.output_ids, ref))

    def test_stop_tokens_and_zero_output(self):
        draft, target, mid = tiny_models()
        prompt = torch.tensor([[1, 2, 3]])
        full = ar_generate(target, prompt, 20, []).generated
        for token in (int(full[0]), int(full[5])):
            ref = ar_generate(target, prompt, 20, [token]).output_ids
            r = reserve_generate(draft, target, mid, prompt, 20, [token])
            self.assertTrue(torch.equal(r.output_ids, ref))
        r = reserve_generate(draft, target, mid, prompt, 0, [])
        self.assertTrue(torch.equal(r.output_ids, prompt))

    def test_snapshots_are_not_mutated_and_alternative_features_are_valid(self):
        draft, target, mid = tiny_models()
        b = TorchReserveBackend(draft, target, mid, max_length=24,
                                reserve_size=2, fork_threshold=1.0)
        state = b.prefill((1, 2, 3))
        f = state.payload
        original = f.feats.clone()
        job = b.draft(state)
        states = b.middle(job)
        self.assertGreater(len(states), 1)
        self.assertEqual(f.mcache.get_seq_length(), 3)
        self.assertEqual(f.dcache.get_seq_length(), 0)
        self.assertTrue(torch.equal(original, f.feats))
        with torch.inference_mode():
            for s in states:
                out = mid(torch.tensor([s.prefix[:-1]]), output_hidden_states=True)
                expected = extract_context_feature(out.hidden_states, draft.target_layer_ids)
                self.assertTrue(torch.allclose(expected, s.payload.feats, atol=1e-5))
                self.assertEqual(s.payload.mcache.get_seq_length(), len(s.prefix) - 1)
                # Every alternate anchor can be drafted without a new middle forward.
                self.assertTrue(b.draft(s).tokens)

    def test_benchmark_writes_matched_records(self):
        from scripts import exp9_reserve
        draft, target, mid = tiny_models()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.json"
            argv = ["exp9_reserve", "--target-device", "cpu", "--mid-device", "cpu",
                    "--draft-device", "cpu", "--prompt", "local fixture", "--max-new", "8",
                    "--repeats", "1", "--reserves", "0", "1", "--trace", "--output", str(output)]
            with (patch("sys.argv", argv),
                  patch.object(exp9_reserve, "load_target", return_value=target),
                  patch.object(exp9_reserve, "load_mid", return_value=mid),
                  patch.object(exp9_reserve, "load_draft", return_value=draft),
                  patch.object(exp9_reserve, "load_tokenizer", return_value=None),
                  patch.object(exp9_reserve, "stop_ids", return_value=[]),
                  patch.object(exp9_reserve, "encode", return_value=torch.tensor([[1, 2, 3]]))):
                exp9_reserve.main()
            records = json.loads(output.read_text())["records"]
            self.assertEqual({r["config"] for r in records}, {"sync", "reserve-0", "reserve-1"})
            self.assertTrue(all(r["match"] and r["timed_tokens"] == 7 for r in records))
            self.assertTrue(all(r["trace"] and r["confirmed_tok_s"] > 0 for r in records))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_two_device_execution(self):
        draft, target, mid = tiny_models()
        draft, target, mid = draft.to("cuda:1"), target.to("cuda:0"), mid.to("cuda:1")
        prompt = torch.tensor([[1, 2, 3]], device="cuda:0")
        ref = ar_generate(target, prompt, 24, []).output_ids
        r = reserve_generate(draft, target, mid, prompt, 24, [],
                             ReserveConfig(2, target_fallback=False), fork_threshold=1.0)
        self.assertTrue(torch.equal(r.output_ids, ref))

    @unittest.skipUnless(torch.cuda.device_count() >= 3, "requires three CUDA devices")
    def test_three_device_execution(self):
        draft, target, mid = tiny_models()
        draft, target, mid = draft.to("cuda:2"), target.to("cuda:0"), mid.to("cuda:1")
        prompt = torch.tensor([[1, 2, 3]], device="cuda:0")
        ref = ar_generate(target, prompt, 24, []).output_ids
        r = reserve_generate(draft, target, mid, prompt, 24, [],
                             ReserveConfig(2, target_fallback=False), fork_threshold=1.0)
        self.assertTrue(torch.equal(r.output_ids, ref))


if __name__ == "__main__":
    unittest.main()
