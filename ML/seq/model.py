import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_p)
        
        # (Batch, Seq_Len, Input_Size) -> (Batch, Seq_Len, Hidden_Size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p if num_layers > 1 else 0)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Size)
        output, (hidden, cell) = self.lstm(x)
        return output, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (Batch, Hidden_Size) 
        # encoder_outputs: (Batch, Seq_Len, Hidden_Size)
        
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state seq_len times
        # (Batch, Seq_Len, Hidden_Size)
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
        
        # (Batch, Seq_Len, 1) -> (Batch, Seq_Len)
        attention_scores = self.v(energy).squeeze(2)
        
        return torch.softmax(attention_scores, dim=1)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_p)
        self.attention = Attention(hidden_size)
        
        self.lstm = nn.LSTM(hidden_size + output_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_val, hidden, cell, encoder_outputs):
        # input_val: (Batch, 1, Output_Size) -> One step at a time
        # hidden: (Num_Layers, Batch, Hidden_Size)
        
        # (Batch, Seq_Len)
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        
        # (Batch, 1, Seq_Len) bmm (Batch, Seq_Len, Hidden_Size) -> (Batch, 1, Hidden_Size)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)
        
        # (Batch, 1, Output_Size + Hidden_Size)
        rnn_input = torch.cat((input_val, context), dim=2)
        
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        # (Batch, 1, Hidden_Size * 2) -> (Batch, 1, Output_Size)
        output = self.out(torch.cat((output, context), dim=2))
        
        return output, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        # src: (Batch, Seq_Len, Input_Size)
        # tgt: (Batch, Seq_Len) - Indices (Optional)
        
        batch_size = src.shape[0]
        tgt_vocab_size = self.decoder.output_size
        
        # Determine sequence length
        if tgt is not None:
            tgt_len = tgt.shape[1]
        else:
            tgt_len = src.shape[1] 
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Initialize decoder input (Batch, 1, Output_Size) with zeros
        decoder_input = torch.zeros(batch_size, 1, tgt_vocab_size).to(self.device)
        
        # Pre-calculate One-Hot for teacher forcing if tgt is provided
        tgt_one_hot = None
        if tgt is not None:
            # Ensure tgt indices are within vocab range
            safe_tgt = tgt.clone()
            safe_tgt[safe_tgt >= tgt_vocab_size] = tgt_vocab_size - 1
            safe_tgt[safe_tgt < 0] = 0
            tgt_one_hot = F.one_hot(safe_tgt, num_classes=tgt_vocab_size).float()
        
        for t in range(tgt_len):
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output.squeeze(1)
            
            # Teacher forcing: decide whether to use actual target or predicted output
            teacher_force = random.random() < teacher_forcing_ratio if tgt is not None else False
            
            if teacher_force and t < tgt_len - 1:
                # Use ground truth (One-Hot Encoded)
                decoder_input = tgt_one_hot[:, t, :].unsqueeze(1)
            else:
                # Auto-regressive: Use softmax probability distribution as continuous input for next step
                decoder_input = torch.softmax(output, dim=2)
                
        predictions = outputs.argmax(dim=2)
        return outputs, predictions

def build_seq2seq(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, device='cpu'):
    enc = EncoderRNN(input_dim, hidden_dim, num_layers, dropout)
    dec = DecoderRNN(hidden_dim, output_dim, num_layers, dropout)
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    # Init weights
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
            
    return model
