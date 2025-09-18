# smallm

Smallm (smaLL + LLm) is my attempt on making a tiny toy language model just for fun and educational purposes. It has about 17.9m parameters and is trained on roughly 3.25 billion tokens of the Cosmopedia dataset. This is very small compared to LLMs' standards, which also explains why it is kinda goofy when you use it (lol), but you can definitely train this on a mid-range card for just half a day or 1-2 days, and it can still generate proper English and data that should be related to the user's prompt.

## Setup

Setup venv and install necessary packages:

```sh
# Create and activate venv (run this every time you start)
python -m venv venv
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
python main.py
```

A prompt will appear for you to chat with the model. You can also import the `ChatBot` class for more control if needed.

## Training

Head over to `./main.py` and change `training` to `True`, then run:
```sh
python main.py
```

The model will train with 3.25b tokens with 10 325m-token segments (estimated 18-20 hours on my Laptop RTX 5070), and after each epoch it will save the current model to `./chatbot.pth`.

To start from where you left off, just name your file `chatbot_continue.pth` to resume training.

## Architecture

Currently it uses:

* Tokenizer: Tiktoken with GPT-2 encoding (50,257 vocab size)
* Embedding: 256-dimensional token embeddings
* Positional Encoding: 256-dimensional position embeddings
* Transformer: 6 encoder layers, 8 attention heads, 1024 d_ffn, 256 d_model
* Output: Linear layer to vocabulary

and is trained with:

* Dataset: Cosmopedia (~3.25b tokens) with 50% overlapping
* Context Window: 1024 tokens
* Batch Size: 8 (effective batch size: 64 with gradient accumulation)
* Optimizer: AdamW with mixed precision training

and generates text with:

* Sampling: Top-k sampling (k=50)
* Temperature: 0.8
* Context Window: 1024 tokens
* Stopping: EOS token for fixed limit (10240 by default)
* Simple repetition penalty

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
