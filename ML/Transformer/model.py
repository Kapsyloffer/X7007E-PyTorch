import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- NEW IMPORT
import math

class InputEmbeddings(nn.Module):
    
    #d_model = dimensioner
    def __init__(self, d_model: int, input_dim: int):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        # Replaced Embedding with Linear for continuous features
        self.projection = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.projection(x) * math.sqrt(self.d_model)
        # In the embedding layers, we multiply those weights by sqrt(d_model)

class PositionalEncoding(nn.Module):

    # Dropout hjälper med att reducera overfitting
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # Dictionary unhooked. 
    
    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        seq_length = x.size(1)
        device = x.device
        
        #pe = PositionalEncoding (based on input length)
        pe = torch.zeros(seq_length, self.d_model, device=device)
        
        # Represents the position of the model inside the sequence
        position = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1) #(Seq_len, 1) <-- Tensor
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * -math.log(10000.0) / self.d_model)
        
        #Apply the sin to the even positions (cos to uneven)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # ^ x::y, start at x, go forward by y

        #Batch dimension to "sentence"
        pe = pe.unsqueeze(0) 

        x = x + (pe).requires_grad_(False) # Not learned tensor
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    #epsilon: Really small number
    def __init__(self, eps:float = 1E-6):
        super().__init__()
        self.eps = eps
        # Parameters = Learnable
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.L1 = nn.Linear(d_model, d_ff) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.L2 = nn.Linear(d_ff, d_model) #W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.L2(self.dropout(torch.relu(self.L1(x))))

# Q x WQ = Q'
# K x WK = K'
# V x WV = V'
# Then split them all in x sizes, heads. 
# They split along embedding dim, not sequence dim
# Each head has a access to the full sentence, but different embeddings
# Attention(Q, K, V) -> softmax each pieces
# Then concat and multiply by W0
# Which results in MHA
class MultiHeadAttention(nn.Module):
    #h = num heads
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model not divisible by h"

        self.d_k = d_model //h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # Flash Attention 
        if hasattr(F, "scaled_dot_product_attention"):
             # query shape: (Batch, h, seq, d_k)
             # mask shape: (B, 1, 1, S) or (1, S, S) usually works.
             
             return F.scaled_dot_product_attention(
                 query, key, value,
                 attn_mask=mask if mask is not None else None,
                 dropout_p=dropout.p if dropout else 0.0,
                 is_causal=False 
             ), None
        
        # --- Fallback for old PyTorch (Memory Heavy) ---
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
        #(For model, for visualizing)
    
    # mask if we want some words to not interact with other words
    def forward(self, q, k, v, mask):
        
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        query = self.w_q(q) 
 
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        key = self.w_k(k) 
 
        # (Batch, Seq_Len, d_model) -> (Batch, Seq_Len, d_model)
        value = self.w_v(v) 

        #(Batch, Seq_Len, d_model) -> (Batch, Seq_Len, h, d_k) -> (Batch, h, seq_len, d_k)
        #We want each head to watch (seq_len, d_k)
        # Full sentence, smaller embedding
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        #(Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, feed_forward, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Combine the feed forward and x, then apply the residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    #self attention eftersom samma värde är key value och query
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    # cross attention då vi blandar sources

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class PointerHead(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)

    def forward(self, decoder_output, encoder_output):
            # decoder_output: (Batch, tgt_len, d_model)
            # encoder_output: (Batch, src_len, d_model)
            
            # Project both to point space
            q = self.W_q(decoder_output)
            k = self.W_k(encoder_output)
            
            # Pointer Attention Score
            # (Batch, tgt_len, d_model) @ (Batch, d_model, src_len) -> (Batch, tgt_len, src_len)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            
            return scores # Raw logits for CrossEntropyLoss

class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, pointer_head: PointerHead, input_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.pointer_head = pointer_head
        
        self.sos_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, dec_out, enc_out):
        return self.pointer_head(dec_out, enc_out)

    def forward(self, src, tgt_indices=None, src_mask=None, tgt_mask=None):
        # src: [Batch, Seq_Len, Input_Dim]
        # tgt_indices: [Batch, Seq_Len] (Indices pointing to sorted src) for Teacher Forcing
        
        encoder_output = self.encode(src, src_mask)
        
        if tgt_indices is not None:
            # src is (B, S, F), indices (B, T). Target (B, T, F)
            B, S, F = src.shape
            expanded_indices = tgt_indices.unsqueeze(-1).expand(-1, -1, F)
            tgt_features = torch.gather(src, 1, expanded_indices)
            sos = self.sos_token.expand(B, 1, -1)
            decoder_input_features = torch.cat([sos, tgt_features[:, :-1, :]], dim=1)
        else:
            B, S, F = src.shape
            decoder_input_features = self.sos_token.expand(B, 1, -1)

        decoder_output = self.decode(encoder_output, src_mask, decoder_input_features, tgt_mask)

        # [batch_size, tgt_len, src_len] 
        output = self.project(decoder_output, encoder_output)
        return output

def build_transformer(input_dim: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff = 2048) -> Transformer:
    # N = number of blocks
    # h = number of heads

    #embedding layers using input_dim instead of vocab_size
    src_embed = InputEmbeddings(d_model, input_dim)
    tgt_embed = InputEmbeddings(d_model, input_dim)

    src_pos = PositionalEncoding(d_model, dropout)
    tgt_pos = PositionalEncoding(d_model, dropout) 

    #Create the encoder and decoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    
    #Create encoder and decoder

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection layer is now replaced by PointerHead
    pointer_head = PointerHead(d_model)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, pointer_head, input_dim)

    #initialize parameters 
    for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return transformer
