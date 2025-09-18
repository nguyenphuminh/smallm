import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken
from datasets import load_dataset
from torch.amp import autocast, GradScaler
import os
import random

class ChatBot(nn.Module):
    def __init__(self, options={}):
        super().__init__()
        
        # Vocab setup - tiktoken BPE from GPT2
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = options.get("vocab_size", 50257)
        self.eos_token_id = self.encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        # Config
        self.d_model = options.get("d_model", 256)
        self.num_layers = options.get("num_layers", 6)
        self.num_heads = options.get("num_heads", 8)
        self.dropout_rate = options.get("dropout_rate", 0.1)
        self.max_seq_len = options.get("max_seq_len", 1024)
        self.overlapping = options.get("overlapping", 2)
        
        # Layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # One-hot output
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Initialize custom weights
        self.apply(self._init_weights)

        # Tie weights
        self.output.weight = self.embedding.weight
        
        # Only use CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but required")
        self.device = torch.device("cuda")
        self.to(self.device)

        # Causal mask cache for different seq_len
        self.mask_cache = {}
        # Position cache
        self.pos_cache = {}
    
    def _init_weights(self, module):
        # Small weights for stable training
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # [1, seq_len]
        
        batch_size, seq_len = token_ids.size()
        
        # Generate causal mask
        if seq_len not in self.mask_cache:
            tgt_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=self.device), diagonal=1)
            self.mask_cache[seq_len] = tgt_mask
        else:
            tgt_mask = self.mask_cache[seq_len]
        
        # Token embedding and positional embedding
        embedding = self.embedding(token_ids)  # [batch_size, seq_len, d_model]

        if seq_len not in self.pos_cache:
            pos = torch.arange(0, seq_len, dtype=torch.int32, device=self.device)
            self.pos_cache[seq_len] = pos
        else:
            pos = self.pos_cache[seq_len]

        pos_emb = self.pos_embedding(pos)
        embedding = self.dropout(embedding + pos_emb)
        
        # Transformer forward pass
        output = self.transformer(embedding, mask=tgt_mask)  # [batch_size, seq_len, d_model]
        output = self.output(output)  # [batch_size, seq_len, vocab_size]
        
        return output
    
    def train_model(self, data_loader, sequence_length=1024, batch_size=8, gradient_accumulation_steps=8):
        print(f"Training with batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

        # Cap context window
        sequence_length = min(sequence_length, self.max_seq_len)
        
        # Init training utils
        optimizer = optim.AdamW(self.parameters(), lr=0.0002, fused=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(self.device.type)

        for segment_index, segment in enumerate(data_loader()):
            # Encode segment to tokens
            tokens = self.text_to_tokens(segment)
            print(f"Segment {segment_index + 1}: {len(segment)} chars -> {len(tokens)} tokens")
            
            # Pre-create all sequences 
            sequences = []
            for start_idx in range(0, len(tokens) - sequence_length, sequence_length // self.overlapping):
                sequence = tokens[start_idx:start_idx + sequence_length]
                if len(sequence) == sequence_length:
                    sequences.append(sequence)
            
            print(f"Segment {segment_index + 1}: Pre-computed {len(sequences)} sequences in memory")

            # Training loop for this segment
            self.train()

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

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            current_lr = optimizer.param_groups[0]["lr"]

            # Log and save
            print(f"Segment {segment_index + 1}: Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Batches: {num_batches}")
            self.save()
            print(f"Segment {segment_index + 1}: Saved to chatbot.pth")

    def generate(
        self,
        prompt,
        context_window=1024,
        max_length=10240,
        repetition_penalty=1.1,
        repetition_penalty_range=64,
        temperature=0.8,
        topk=50
    ):
        print(prompt)

        self.eval()
        
        with torch.no_grad():
            prompt_tokens = self.text_to_tokens(prompt)
            current_tokens = prompt_tokens.copy()
            max_context = min(context_window, self.max_seq_len)

            # Stack in case a char is made up of multiple tokens
            word_stack = []

            for i in range(max_length):
                current_tokens = current_tokens[-max_context:] if len(current_tokens) > max_context else current_tokens
                input_tensor = torch.tensor(current_tokens, device=self.device).unsqueeze(0)

                # Forward pass
                output = self.forward(input_tensor)
                logits = output[0, -1, :]

                # Apply temperature scaling
                scaled_logits = logits / temperature

                # Repetition penalty
                if len(current_tokens) > 0:
                    # Count frequency of each token in recent context
                    recent_tokens = current_tokens[-repetition_penalty_range:]
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
                top_k_values, top_k_indices = torch.topk(scaled_logits, k=topk)
                top_k_probs = torch.softmax(top_k_values, dim=0)

                # Sample from top-k
                sampled_index = torch.multinomial(top_k_probs, 1).item()
                next_token_id = top_k_indices[sampled_index].item()

                # Push newest token
                current_tokens.append(next_token_id)

                # Stream output
                word_stack.append(next_token_id)
                decoded_word = self.tokens_to_text(word_stack)

                if "\ufffd" not in decoded_word:
                    print(decoded_word, end="")
                    word_stack = []

                # Stop on eos token
                if next_token_id == self.eos_token_id:
                    break
                
                # Clear caches
                self.mask_cache.clear()
                self.pos_cache.clear()
                torch.cuda.empty_cache()
    
    def save(self, path="./chatbot.pth"):
        torch.save({
            "model_state_dict": self.state_dict(),
            "d_model": self.d_model,
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
def load_data(
    start=0,
    end=3000000,
    step=300000,
    path="HuggingFaceTB/cosmopedia",
    name="web_samples_v2",
    split="train"
):
    # Init dataset loader
    dataset = load_dataset(path, name, split=split)
    dataset_size = len(dataset)

    # Load small segments of the dataset
    for segment_index in range(start, end, step):
        if segment_index >= dataset_size:
            return

        segment_end = min(segment_index + step, dataset_size, end)
        miniset = dataset.select(range(segment_index, segment_end))
        text_parts = []

        for sample_index, sample in enumerate(miniset):
            prompt = sample.get("prompt", "")
            text = sample.get("text", "")
            if prompt and text:
                text_parts.append(f"Human: {prompt}\n\nAssistant: {text}<|endoftext|>\n\n")

        combined_text = "".join(text_parts)

        yield combined_text

if __name__ == "__main__":
    # Training flag for convenience
    training = False

    # Initialize model, full vocab, ~17.9m params
    chatbot = ChatBot()

    print(f"Using device: {chatbot.device}")
    print(f"Model parameters: {sum(p.numel() for p in chatbot.parameters()):,}")

    if training:
        # Load existing model to continue training if exists
        if os.path.exists("./chatbot_continue.pth"):
            print("Found model to continue training from")
            chatbot.load("./chatbot_continue.pth")

        # Train
        chatbot.train_model(load_data)
        print("Final save to chatbot.pth")
        chatbot.save()
    else:
        # Load
        print("Loaded from chatbot.pth")
        chatbot.load()

        # Prompt
        while True:
            prompt = input("\n\n\033[32mPrompt (type \"/help\" to see a list of commands): ")

            print("\033[33m")

            if prompt == "/help":
                print("All commands:")
                print("/help - View a list of commands")
                print("/clear - Clear the console")
            elif prompt == "/clear":
                print("\033c", end="")
            else:
                chatbot.generate(f"Human: {prompt}\n\nAssistant:")
                print("\033[0m")
