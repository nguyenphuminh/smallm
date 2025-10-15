# PlanckGPT

PlanckGPT is my attempt on making a tiny language model (planck length refence :D) from scratch mostly for fun and educational purposes, but also to see how far a consumer-level computer can go. It has about 150m parameters and is trained on roughly 3 billion tokens of the Fineweb dataset. This is small compared to modern LLMs' standards, which also explains why it is goofy when you use it (lol), but you can definitely train this on a mid-range card for just 3-4 days, and it can still generate proper English and data that should be related to the user's prompt.

## Setup

Setup venv and install necessary packages:

```sh
# Create and activate venv
python -m venv venv
# Run this every time you start
source venv/scripts/activate
# or "./venv/scripts/activate" if you are on windows

# Install packages (once)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install tiktoken datasets
```

Of course, you should already install compatible CUDA and Python versions, I currently use Python 3.13 and CUDA 13 (which is compatible with CUDA 12.8 mentioned above).

## Running PlanckGPT

1. Download the latest model (`chatbot.pth`) in the releases page.
2. Simply run:
```sh
python inference.py
```

A prompt will appear for you to chat with the model.

## Training

To train the model from scratch, run:
```sh
python train.py
```

The model will train with 3b+ tokens with 20 150m-token segments (estimated 48 hours on my Laptop RTX 5070), and after each epoch it will save the current model to `./chatbot.pth`.

## Architecture

Currently it uses:

* Tokenizer: Tiktoken with GPT-2 encoding (50,257 vocab size).
* Embedding: 768-dimensional token embedding.
* Rotary positional embedding.
* Transformer: 12 decoder layers, 12 heads, 3072 d_ffn, 768 d_model.
* Multi-Query Attention with flash attention support (sdpa).
* Squared ReLU for activation.
* RMSNorm without learnable params for normalization, used in transformer and before output.
* Output: Linear layer to vocabulary (without weight-tying).

and is trained with:

* Dataset: Fineweb (~3b tokens) with no overlapping.
* Context Window: 1024 tokens.
* Batch Size: 8 (effective batch size: 512 with gradient accumulation).
* Muon optimizer for transformer weights, fused AdamW optimizer for embedding and linear layers.
* LinearLR for 2% warmup, CosineAnnealingLR for lr decay.
* BF16 mixed precision training and other Blackwell-specific features.
* Training with torch.compile on "max-autotune" mode.

and generates text with:

* Sampling: Top-k sampling (k=50).
* Temperature: 0.7.
* Context Window: 1024 tokens.
* Stopping: EOS token for fixed limit (10240 by default).
* Simple repetition penalty with 64 latest tokens.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the Apache 2.0 License.
