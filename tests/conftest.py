"""Offline fixtures: a tiny Llama-style model + tokenizer saved to disk, and fake corpus loaders.

Nothing here touches the network, so the whole suite runs on CPU in CI.
"""

from __future__ import annotations

import random

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from lfm3_cpt.config import Config
from lfm3_cpt.data import Loaders

WORDS = [
    "ev",
    "evler",
    "evlerde",
    "kitap",
    "kitaplar",
    "okul",
    "okula",
    "okuldan",
    "gel",
    "geldi",
    "gidiyor",
    "güzel",
    "büyük",
    "küçük",
    "şehir",
    "türkiye",
    "istanbul",
    "ankara",
    "tarih",
    "nehir",
    "dağ",
    "deniz",
    "ve",
    "bir",
    "bu",
    "çok",
    "ile",
    "için",
    "olarak",
    "yılında",
    "kurulan",
]


def fake_text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words)) + "."


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("tiny-model")
    rng = random.Random(0)
    corpus = [fake_text(rng, 40) for _ in range(300)]

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)
    from tokenizers import decoders

    tok.decoder = decoders.ByteLevel()
    # Like LFM's tokenizer: prepend BOS to every encoded text.
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", tok.token_to_id("<s>"))]
    )
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>"
    )
    hf_tok.save_pretrained(path)

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(hf_tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        bos_token_id=hf_tok.bos_token_id,
        eos_token_id=hf_tok.eos_token_id,
        pad_token_id=hf_tok.pad_token_id,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tokenizer(tiny_model_dir):
    from lfm3_cpt.modeling import load_tokenizer

    return load_tokenizer(tiny_model_dir, trust_remote_code=False)


@pytest.fixture()
def model(tiny_model_dir):
    from lfm3_cpt.modeling import load_model

    return load_model(tiny_model_dir, torch.float32, device_map=None, trust_remote_code=False)


def make_wiki(n: int = 60, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # every 4th article is a stub that must be filtered out
        words = 5 if i % 4 == 3 else rng.randint(60, 120)
        out.append({"title": f"Madde {i}", "text": fake_text(rng, words)})
    return out


def make_grammar(per: int = 3) -> Dataset:
    rows = {"sentence_good": [], "sentence_bad": [], "linguistic_phenomenon": []}
    for p in ("binding", "passives", "scrambling"):
        for i in range(per):
            rows["sentence_good"].append(f"bu ev çok güzel {i}.")
            rows["sentence_bad"].append(f"bu evler çok güzel {i}.")
            rows["linguistic_phenomenon"].append(p)
    return Dataset.from_dict(rows)


@pytest.fixture()
def fake_loaders() -> Loaders:
    rng = random.Random(2)
    return Loaders(
        wiki=lambda cfg: iter(make_wiki()),
        ood=lambda cfg: iter([{"text": ""}] + [{"text": fake_text(rng, 50)} for _ in range(20)]),
        grammar=lambda cfg: make_grammar(),
    )


@pytest.fixture()
def tiny_config(tmp_path, tiny_model_dir) -> Config:
    from lfm3_cpt.config import build_config

    return build_config(
        "cpu-debug",
        overrides={
            "model_id": tiny_model_dir,
            "trust_remote_code": False,
            "output_dir": str(tmp_path / "run"),
            "data_dir": str(tmp_path / "data"),
            "train_articles": 30,
            "eval_articles": 4,
            "ood_pages": 4,
            "min_chars": 100,
            "max_length": 64,
            "max_steps": 3,
            "lora_r": 4,
            "lora_alpha": 8,
            "grammar_batch_size": 4,
            "save_steps": 2,
            "phenomena": ["binding", "passives", "scrambling"],
        },
    )
