# smallm

Smallm (smaLL + LLm) is my attempt on making a tiny toy language model from scratch just for fun and educational purposes. It has about 20m parameters and is trained on roughly 10 billion tokens of the Fineweb dataset. This is very small compared to LLMs' standards, which also explains why it is goofy when you use it (lol), but you can definitely train this on a mid-range card for just half a day or 1-2 days, and it can still generate proper English and data that should be related to the user's prompt.

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

## Running smallm

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

The model will train with 3.25b tokens with 10 325m-token segments (estimated 17 hours on my Laptop RTX 5070), and after each epoch it will save the current model to `./chatbot.pth`.

## Architecture

Currently it uses:

* Tokenizer: Tiktoken with GPT-2 encoding (50,257 vocab size).
* Embedding: 256-dimensional token embeddings.
* Positional Encoding: 256-dimensional position embeddings.
* Transformer: 8 decoder layers, 6 heads, 1024 d_ffn, 256 d_model.
* Grouped Query Attention with 1 kv head and flash attention (sdpa).
* PaLM-style parallel attention output.
* RMSNorm rather than LayerNorm for normalization.
* SwiGLU activation in the FFN layers.
* Output: Weight-tied linear layer to vocabulary.

and is trained with:

* Dataset: Fineweb (~10b tokens) with no overlapping.
* Context Window: 256 tokens.
* Batch Size: 31 (effective batch size: 248 with gradient accumulation)
* Optimizer: AdamW with mixed precision training
* LinearLR for 2% warmup, CosineAnnealingLR for lr decay.

and generates text with:

* Sampling: Top-k sampling (k=50)
* Temperature: 0.7
* Context Window: 256 tokens
* Stopping: EOS token for fixed limit (10240 by default)
* Simple repetition penalty

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
