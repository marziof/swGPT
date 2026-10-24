
# swGPT: small GPT from scratch 

Goal: 
To build a small GPT-style language model built from scratch in PyTorch: pretrained on
TinyStories, then finetuned on Star Wars (IV-VI) dialogue, then improved with RL/RLHF.
The aim is to understand each element (tokenizer, RoPE, attention, KV cache, optimizers,
evaluation, RL) by implementing it directly.

**Status:** work in progress. Data pipeline done; model next.

## Setup

```bash
pip install -e .
pip install datasets tokenizers numpy torch
```

## Data

Download TinyStories, train a byte-level BPE tokenizer, and write tokenized
`train.bin` / `val.bin` (raw `uint16`) to `data/`:

```bash
# quick local dev run (streams, no full download)
python scripts/prepare_data.py --streaming --num_stories 10000

# full run (e.g. on Kaggle)
python scripts/prepare_data.py --num_stories 500000
```

Outputs in `data/`: `tokenizer.json`, `train.bin`, `val.bin`, `meta.json`.
The tokenizer is trained once and reused; pass `--overwrite` to retrain it.
Read the binaries with `np.memmap(path, dtype=np.uint16, mode="r")`.

`data/` is gitignored. The Star Wars dialogue data is copyrighted: keep it private.

## Planned architecture

- Byte-level BPE (~4k vocab) with reserved special tokens
- Token embedding tied with the output projection
- RoPE applied to Q and K inside attention
- Pre-norm (RMSNorm) decoder blocks: causal multi-head attention + MLP
- AdamW baseline, Muon as an ablation
- KV cache for generation

## Roadmap

- [x] Data prep and batch loader
- [ ] Config and project plumbing
- [ ] Model (RMSNorm, RoPE, attention, MLP, block, GPT)
- [ ] Training loop, eval, checkpoint/resume
- [ ] W&B logging and Kaggle runs
- [ ] Generation and KV cache
- [ ] Evaluation suite (perplexity, bits per char, distinct-n, memorization)
- [ ] Ablations (Muon vs AdamW, RoPE vs learned positions)
- [ ] SFT on Star Wars dialogue
- [ ] RL / RLHF / DPO