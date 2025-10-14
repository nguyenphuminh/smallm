import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

class MultiQueryAttention(nn.Module):
    # MQA with Flash Attention - maximum memory efficiency
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, 1, self.head_dim).transpose(1, 2)
        
        # Expand KV to match Q heads
        k = k.expand(B, self.num_heads, L, self.head_dim)
        v = v.expand(B, self.num_heads, L, self.head_dim)
        
        # Flash Attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        return self.out_proj(out.transpose(1, 2).reshape(B, L, -1))

class OptimizedTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, dim_ff):
        super().__init__()

        self.attn = MultiQueryAttention(dim, num_heads)
        self.ffn1 = nn.Linear(dim, dim_ff, bias=False)
        self.ffn2 = nn.Linear(dim_ff, dim, bias=False)
    
    def forward(self, x):
        x = x + self.attn(rms_norm(x))
        x = x + self.ffn2(F.relu(self.ffn1(rms_norm(x))))
        return x

class ChatBot(nn.Module):
    def __init__(self, options={}):
        super().__init__()
        
        # Vocab setup - tiktoken BPE from GPT2
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = options.get("vocab_size", 50257)
        self.eos_token_id = self.encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        # Config
        self.d_model = options.get("d_model", 768)
        self.num_layers = options.get("num_layers", 12)
        self.num_heads = options.get("num_heads", 12)
        self.max_seq_len = options.get("max_seq_len", 1024)
        self.overlapping = options.get("overlapping", 1)
        
        # Layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)

        # Transformer decoder layers
        self.transformer = nn.ModuleList([
            OptimizedTransformerLayer(
                self.d_model,
                self.num_heads,
                self.d_model * 4
            ) for _ in range(self.num_layers)
        ])

        # One-hot output
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Apply weight init
        self.apply(self._init_weights)

        # Tie weights
        # self.output.weight = self.embedding.weight
        
        # Only use CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but required")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.device = torch.device("cuda")
        self.to(self.device)

        # Position cache
        self.pos = torch.arange(0, self.max_seq_len, dtype=torch.long, device=self.device)
    
    def _init_weights(self, module):
        # Small weights for stable training
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids):        
        # Token embedding and positional embedding
        seq_len = token_ids.size(1)
        embedding = self.embedding(token_ids)
        pos_emb = self.pos_embedding(self.pos[:seq_len])
        embedding = embedding + pos_emb
        
        # Transformer forward pass
        for layer in self.transformer:
            embedding = checkpoint(layer, embedding, use_reentrant=False)

        # Final norm
        embedding = rms_norm(embedding)

        # Linear output projection
        output = self.output(embedding)
        
        return output
    
    def train_model(self, data_loader, sequence_length=1024, batch_size=8, gradient_accumulation_steps=64, lr=0.0002, T_max=5722):
        print(f"Training with batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

        # Cap context window
        sequence_length = min(sequence_length, self.max_seq_len)
        
        # Init training utils
        optimizer = optim.AdamW(self.parameters(), lr=lr, fused=True)
        warmup_steps = int(0.02 * T_max)  # 2% warmup
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max-warmup_steps, eta_min=1e-5)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        criterion = nn.CrossEntropyLoss()

        for segment_index, segment in enumerate(data_loader):
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
            
            optimizer.zero_grad(set_to_none=True)
            
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
                with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    output = self.forward(input_tokens)  # [batch_size, seq_len-1, vocab_size]
                    output = output.reshape(-1, self.vocab_size)  # [batch_size * seq_len-1, vocab_size]
                    target_tokens = target_tokens.reshape(-1)  # [batch_size * seq_len-1]
                    loss = criterion(output, target_tokens)
                    loss = loss / gradient_accumulation_steps
                
                # Scale loss and backward
                loss.backward()

                total_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                
                # Update weights every gradient_accumulation_steps
                if num_batches % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            # Final update if needed
            if num_batches % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
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
        temperature=0.7,
        topk=50,
        memory=[]
    ):
        print(prompt, end="")

        self.eval()
        
        with torch.no_grad():
            current_tokens = memory + self.text_to_tokens(prompt)
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

                # Stop on eos token and conversation overlap
                if next_token_id == self.eos_token_id or next_token_id == 20490 or next_token_id == 48902:
                    break

                # Push newest token
                current_tokens.append(next_token_id)

                # Stream output
                word_stack.append(next_token_id)
                decoded_word = self.tokens_to_text(word_stack)

                if "\ufffd" not in decoded_word:
                    print(decoded_word, end="")
                    word_stack = []
                
                # Clear caches
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
            "eos_token_id": self.eos_token_id
        }, path)
    
    def load(self, path="./chatbot.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
    
    def text_to_tokens(self, text):
        return self.encoding.encode(text, allowed_special={"<|endoftext|>"})
    
    def tokens_to_text(self, tokens):
        return self.encoding.decode(tokens)
