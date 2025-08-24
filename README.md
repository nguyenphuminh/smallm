# smallm

Smallm (smaLL + LLm) is my attempt on making a tiny toy language model just for fun and educational purposes. It has about 28m parameters and is trained on the Cosmopedia-100k dataset which has roughly 105m tokens. This is very small compared to LLMs' standards, which also explains why it is kinda goofy when you use it (lol), but you can definitely train this on a mid-range card for just half a day or 1-2 days, and it still generates proper English and data that is related to the user's prompt.

## Setup

Setup venv and install necessary packages:

```sh
# Create and activate venv
python -m venv venv
source venv/bin/activate
# or "./venv/bin/activate" if you are on windows

# Install packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install tiktoken datasets
```

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

The model will train for 40 epochs (estimated 10-15 hours on my Laptop RTX 5070). You can adjust configurations in the training method.

To start from where you left off, just name your file `chatbot_continue.pth` to resume training.

## Architecture

Currently it uses:

* Tokenizer: Tiktoken with GPT-2 encoding (50,257 vocab size)
* Embedding: 256-dimensional token embeddings
* Positional Encoding: Sinusoidal positional encoding
* Transformer: 3 encoder layers, 8 attention heads, 256 hidden size
* Output: Linear layer to vocabulary

and is trained with:

* Dataset: Cosmopedia-100k (~105m tokens)
* Context Window: 512 tokens
* Batch Size: 17 (effective batch size: 136 with gradient accumulation)
* Optimizer: AdamW with mixed precision training
* Training Time: ~10-15 hours on RTX 5070

and generates text with:

* Sampling: Top-k sampling (k=50)
* Temperature: 0.8 (configurable)
* Context Window: 512 tokens
* Stopping: Natural EOS token or conversation breaks

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
