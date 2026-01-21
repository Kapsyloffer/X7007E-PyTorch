import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Projector(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.Tanh(),
            nn.LayerNorm(d_model) 
        )
        
    def forward(self, x):
        return self.net(x)

class PointerNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.projector = Projector(input_dim, d_model)
        
        # Bidirectional Encoder: outputs hidden_dim * 2
        self.encoder = nn.LSTM(d_model, hidden_dim, batch_first=True, bidirectional=True)
        
        # Decoder: LSTMCell usually takes (input_size, hidden_size)
        # We feed it the projected encoder output (2 * hidden_dim)
        self.decoder_cell = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        
        # Bridges to map bidirectional encoder states (2*H) to decoder states (H)
        self.encoder_h_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_c_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Glimpse mechanism
        self.glimpse_W1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False) 
        self.glimpse_W2 = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.glimpse_v = nn.Linear(hidden_dim, 1, bias=False)
        
        # Pointer mechanism
        # Fixed: W1 must match the 2*H encoder output
        # Fixed: W2 now maps the combined (H + 2*H) or projected context
        self.ptr_W1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False) 
        self.ptr_W2 = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.ptr_v = nn.Linear(hidden_dim, 1, bias=False)
        
        # New: Project context back to hidden_dim to allow addition with dec_hidden
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self._init_weights()
           
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def attention(self, query, ref, W1, W2, v, mask=None):
        # ref: [Batch, SeqLen, Hidden*2] -> ref_proj: [Batch, SeqLen, Hidden]
        ref_proj = W1(ref)
        
        # query: [Batch, Hidden] -> query_proj: [Batch, 1, Hidden]
        query_proj = W2(query).unsqueeze(1)
        
        # Broadcasting addition: [Batch, SeqLen, Hidden]
        scores = v(torch.tanh(ref_proj + query_proj)).squeeze(-1)
        
        scores = scores / math.sqrt(self.hidden_dim)
        
        if mask is not None:
            scores.masked_fill_(mask, -1e9)
            
        probs = F.softmax(scores, dim=1)
        return scores, probs

    def forward(self, x, targets=None):
        batch_size, seq_len, _ = x.size()
        embedded = self.projector(x)
        
        self.encoder.flatten_parameters()
        encoder_outputs, (hidden, cell) = self.encoder(embedded.contiguous())
        
        # Bridge bidirectional states to single decoder state
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1)
        cell_cat = torch.cat([cell[0], cell[1]], dim=1)
        
        dec_hidden = torch.tanh(self.encoder_h_bridge(hidden_cat))
        dec_cell = torch.tanh(self.encoder_c_bridge(cell_cat))
        
        # Initial decoder input: mean of encoder outputs [B, H*2]
        decoder_input = torch.mean(encoder_outputs, dim=1)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(x.device)
        
        logits_list = []
        pointers_list = []
        
        for t in range(seq_len):
            dec_hidden, dec_cell = self.decoder_cell(decoder_input, (dec_hidden, dec_cell))
            
            # 1. Glimpse
            _, glimpse_probs = self.attention(
                dec_hidden, encoder_outputs, 
                self.glimpse_W1, self.glimpse_W2, self.glimpse_v, 
                mask=mask
            )
            
            # Context: [B, 1, S] @ [B, S, H*2] -> [B, H*2]
            context = torch.bmm(glimpse_probs.unsqueeze(1), encoder_outputs).squeeze(1)
            
            # 2. Pointer Attention
            # Use the context_proj to align context (2*H) with dec_hidden (H)
            combined_query = dec_hidden + torch.tanh(self.context_proj(context))
            
            ptr_scores, _ = self.attention(
                combined_query, encoder_outputs, 
                self.ptr_W1, self.ptr_W2, self.ptr_v, 
                mask=mask
            )
            
            logits_list.append(ptr_scores)
            
            if self.training and targets is not None:
                selected = targets[:, t]
            else:
                probs = F.softmax(ptr_scores, dim=1)
                _, selected = torch.max(probs, dim=1)

            pointers_list.append(selected)

            # Update mask
            chosen_one_hot = torch.zeros_like(ptr_scores, dtype=torch.bool)
            chosen_one_hot.scatter_(1, selected.unsqueeze(1), 1)
            mask = mask | chosen_one_hot
            
            # Next input is the selected encoder output
            gather_idx = selected.view(batch_size, 1, 1).expand(-1, -1, encoder_outputs.size(2))
            decoder_input = torch.gather(encoder_outputs, 1, gather_idx).squeeze(1)
            
        return torch.stack(logits_list, dim=1), torch.stack(pointers_list, dim=1)
