import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken
import json
import os
import math
from datasets import load_dataset
from torch.amp import autocast, GradScaler
import random

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class ChatBot(nn.Module):
    def __init__(self, options):
        super().__init__()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Vocab setup - tiktoken BPE from GPT2
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = options.get("vocab_size", 50257)
        self.eos_token_id = self.encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        # Config
        self.embedding_size = options.get("embedding_size", 256)
        self.hidden_size = options.get("hidden_size", 256)
        self.num_layers = options.get("num_layers", 3)
        self.num_heads = options.get("num_heads", 8)
        self.dropout_rate = options.get("dropout_rate", 0.1)
        self.max_seq_len = options.get("max_seq_len", 1024)
        self.overlapping = options.get("overlapping", 2)
        
        # Layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_encoding = PositionalEncoding(self.hidden_size, self.max_seq_len)
        self.input_projection = nn.Linear(self.embedding_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # One-hot output
        self.output = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Initialize custom weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self, module):
        # Small weights for stable training
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def generate_square_subsequent_mask(self, sz):
        # Generate causal mask for autoregressive generation
        mask = torch.triu(torch.ones(sz, sz, device=self.device) * float("-inf"), diagonal=1)
        return mask
    
    def forward(self, token_ids):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # [1, seq_len]
        
        batch_size, seq_len = token_ids.size()
        
        # Generate causal mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len)
        
        # Embedding and positional encoding
        embedding = self.embedding(token_ids)  # [batch_size, seq_len, embedding_size]
        embedding = self.input_projection(embedding)  # [batch_size, seq_len, hidden_size]
        embedding = self.pos_encoding(embedding)
        embedding = self.dropout(embedding)
        
        # Transformer forward pass
        output = self.transformer(embedding, mask=tgt_mask)  # [batch_size, seq_len, hidden_size]
        output = self.output(output)  # [batch_size, seq_len, vocab_size]
        
        return output
    
    def train_model(self, text, epochs=10, sequence_length=1024, batch_size=8, gradient_accumulation_steps=8, val_text=None):
        # Cap context window
        sequence_length = min(sequence_length, self.max_seq_len)
        
        # AdamW optimizer with learning rate scheduler for stable training
        optimizer = optim.AdamW(self.parameters(), lr=0.0002, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = GradScaler(self.device.type)
        
        tokens = self.text_to_tokens(text)
        print(f"Training data: {len(text)} chars -> {len(tokens)} tokens")
        
        # Pre-create all sequences 
        sequences = []
        for start_idx in range(0, len(tokens) - sequence_length, sequence_length // self.overlapping):
            sequence = tokens[start_idx:start_idx + sequence_length]
            if len(sequence) == sequence_length:
                sequences.append(sequence)
        
        print(f"Pre-computed {len(sequences)} sequences in memory")

        # Prepare validation data if provided
        val_sequences = []
        if val_text:
            val_tokens = self.text_to_tokens(val_text)
            print(f"Validation data: {len(val_text)} chars -> {len(val_tokens)} tokens")
            
            for start_idx in range(0, len(val_tokens) - sequence_length, sequence_length // self.overlapping):
                sequence = val_tokens[start_idx:start_idx + sequence_length]
                if len(sequence) == sequence_length:
                    val_sequences.append(sequence)
            
            print(f"Pre-computed {len(val_sequences)} validation sequences in memory")
        
        print(f"Training with batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

        self.train()

        for epoch in range(epochs):
            # Shuffle all sequences each epoch
            random.shuffle(sequences)

            total_loss = 0
            num_batches = 0
            
            optimizer.zero_grad()
            
            for batch_start in range(0, len(sequences), batch_size):
                batch_sequences = sequences[batch_start:batch_start + batch_size]
                
                # Convert to tensors
                batch_input = []
                batch_target = []
                
                for seq in batch_sequences:
                    batch_input.append(seq[:-1])
                    batch_target.append(seq[1:])
                
                # Stack into tensors [batch_size, seq_len]
                input_tokens = torch.tensor(batch_input, device=self.device)  # [batch_size, seq_len-1]
                target_tokens = torch.tensor(batch_target, device=self.device) # [batch_size, seq_len-1]
                
                # Enable mixed precision
                with autocast(device_type=self.device.type):
                    output = self.forward(input_tokens)  # [batch_size, seq_len-1, vocab_size]
                    output = output.reshape(-1, self.vocab_size)  # [batch_size * seq_len-1, vocab_size]
                    target_tokens = target_tokens.reshape(-1)  # [batch_size * seq_len-1]
                    loss = criterion(output, target_tokens)
                    loss = loss / gradient_accumulation_steps
                
                # Scale loss and backward
                scaler.scale(loss).backward()

                total_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                
                # Update weights every gradient_accumulation_steps
                if num_batches % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)  # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            
            # Final update if needed
            if num_batches % gradient_accumulation_steps != 0:
                scaler.unscale_(optimizer)  # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            scheduler.step()

            # Calculate validation loss if provided
            val_loss = 0
            if val_sequences:
                self.eval()
                with torch.no_grad():
                    val_total_loss = 0
                    val_batches = 0

                    for batch_start in range(0, len(val_sequences), batch_size):
                        batch_sequences = val_sequences[batch_start:batch_start + batch_size]

                        batch_input = []
                        batch_target = []

                        for seq in batch_sequences:
                            batch_input.append(seq[:-1])
                            batch_target.append(seq[1:])

                        input_tokens = torch.tensor(batch_input, device=self.device)
                        target_tokens = torch.tensor(batch_target, device=self.device)

                        output = self.forward(input_tokens)
                        output = output.reshape(-1, self.vocab_size)
                        target_tokens = target_tokens.reshape(-1)
                        loss = criterion(output, target_tokens)

                        val_total_loss += loss.item()
                        val_batches += 1

                    val_loss = val_total_loss / val_batches if val_batches > 0 else 0
                
                self.train()  # Switch back to training mode

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            current_lr = optimizer.param_groups[0]["lr"]

            # Log and save
            if val_sequences:
                print(f"Epoch {epoch + 1}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}, Batches: {num_batches}")
            else:
                print(f"Epoch {epoch + 1}: Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Batches: {num_batches}")
            self.save()
            print(f"Saved to chatbot.pth (epoch {epoch + 1})")

    def generate(
        self,
        prompt,
        context_window=1024,
        max_length=2000,
        repetition_penalty=1.1,
        repetition_penalty_range=64,
        temperature=0.8,
        debug=False
    ):
        print(prompt)

        self.eval()
        
        with torch.no_grad():
            prompt_tokens = self.text_to_tokens(prompt)
            result_tokens = prompt_tokens.copy()

            # Stack in case a char is made up of multiple tokens
            word_stack = []

            if debug:
                print(f"Prompt tokens: {prompt_tokens[:10]}...")
                print(f"EOS token ID: {self.eos_token_id}")

            for i in range(max_length):
                max_context = min(context_window, self.max_seq_len)
                context_tokens = result_tokens[-max_context:] if len(result_tokens) > max_context else result_tokens

                input_tensor = torch.tensor(context_tokens, device=self.device).unsqueeze(0)

                # Forward pass
                output = self.forward(input_tensor)
                logits = output[0, -1, :].cpu()

                # Apply temperature scaling
                scaled_logits = logits / temperature

                # Repetition penalty
                if len(result_tokens) > 0:
                    # Count frequency of each token in recent context
                    recent_tokens = result_tokens[-repetition_penalty_range:]
                    token_counts = {}
                    for token in recent_tokens:
                        token_counts[token] = token_counts.get(token, 0) + 1
                    
                    # Apply penalty based on frequency
                    for token_id, count in token_counts.items():
                        if token_id < len(scaled_logits):
                            # Penalty increases with frequency
                            penalty = repetition_penalty ** count
                            if scaled_logits[token_id] > 0:
                                scaled_logits[token_id] /= penalty
                            else:
                                scaled_logits[token_id] *= penalty

                # Top-k sampling
                top_k_values, top_k_indices = torch.topk(scaled_logits, k=50)
                top_k_probs = torch.softmax(top_k_values, dim=0)

                # Sample from top-k
                sampled_index = torch.multinomial(top_k_probs, 1).item()
                next_token_id = top_k_indices[sampled_index].item()

                result_tokens.append(next_token_id)

                # Stream output
                word_stack.append(next_token_id)
                decoded_word = self.tokens_to_text(word_stack)

                if "\ufffd" not in decoded_word:
                    print(decoded_word, end="")
                    word_stack = []

                if debug and i < 10:
                    decoded = self.encoding.decode([next_token_id])
                    print(f"Step {i}: token {next_token_id} -> '{decoded}'")

                # Stop on eos token
                if next_token_id == self.eos_token_id:
                    if debug:
                        print(f"Stopped generation at EOS token (step {i})")
                    break

                # Also stop at conversation breaks (backup)
                current_text = self.tokens_to_text(result_tokens)
                if "\n\nHuman:" in current_text or "\n\nUser:" in current_text:
                    break
            
            result = self.tokens_to_text(result_tokens)
            return result
    
    def save(self, path="./chatbot.pth"):
        torch.save({
            "model_state_dict": self.state_dict(),
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "overlapping": self.overlapping,
            "eos_token_id": self.eos_token_id,
            "dropout_rate": self.dropout_rate
        }, path)
    
    def load(self, path="./chatbot.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
    
    def text_to_tokens(self, text):
        return self.encoding.encode(text, allowed_special={"<|endoftext|>"})
    
    def tokens_to_text(self, tokens):
        return self.encoding.decode(tokens)

# Load cosmopedia dataset
def load_cosmopedia(start=0, end=300000):
    print("Loading cosmopedia dataset...")
    
    try:
        dataset = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train")
        dataset = dataset.select(range(start, min(end, len(dataset))))
        
        formatted_texts = []
        for i, sample in enumerate(dataset):
            if i % 1000 == 0:
                print(f"Processing {i}/{len(dataset)}")

            text = sample.get("text", "")
            prompt = sample.get("prompt", "")

            if text and len(text) > 50:
                if prompt:
                    formatted_text = f"Human: {prompt}\n\nAssistant: {text}<|endoftext|>\n\n"
                else:
                    formatted_text = f"Human: Can you explain this?\n\nAssistant: {text}<|endoftext|>\n\n"
                formatted_texts.append(formatted_text)
        
        combined_text = "".join(formatted_texts)
        print(f"\nTotal samples: {len(formatted_texts)}")
        print(f"Total characters: {len(combined_text):,}")
        return combined_text
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    # Training flag for convenience
    training = False

    # Initialize model, full vocab, ~20m params
    chatbot = ChatBot({
        "vocab_size": 50257,
        "embedding_size": 256,
        "hidden_size": 256,
        "num_layers": 3,
        "num_heads": 8,
        "dropout_rate": 0.1,
        "max_seq_len": 1024,
        "overlapping": 2
    })

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")
    print(f"Vocab size: {chatbot.vocab_size} tokens")

    if training:
        # Load data
        CHAT_DATA = load_cosmopedia(0, 300000)
        # Load validation data
        VALIDATION_DATA = load_cosmopedia(300000, 315000)

        # Load existing model to continue training if exists
        if os.path.exists("./chatbot_continue.pth"):
            print("Found model to continue training from")
            chatbot.load("./chatbot_continue.pth")

        # Train
        print("\nTraining transformer for 10 epochs...")
        chatbot.train_model(CHAT_DATA, val_text=VALIDATION_DATA)
        print("Final save to chatbot.pth")
        chatbot.save()

        # Test generation
        print("Test generation:")
        chatbot.generate("Human: How to cook eggs\n\nAssistant:")
    else:
        # Load
        print("Loaded from chatbot.pth")
        chatbot.load()

        # Prompt
        while True:
            prompt = input("\n\n\033[32mPrompt: ")
            print("\033[33m")
            chatbot.generate(f"Human: {prompt}\n\nAssistant:")
            print("\033[0m")