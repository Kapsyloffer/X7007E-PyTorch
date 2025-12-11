import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.vt = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        encoder_outputs, (h, c) = self.encoder(x)

        decoder_input = torch.zeros(batch_size, 1, x.size(2), device=x.device)
        decoder_hidden = (h, c)

        pointers = []

        for _ in range(seq_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            u = self.vt(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_output)))
            attn_scores = u.squeeze(-1)
            attn_probs = F.softmax(attn_scores, dim=1)

            pointers.append(attn_probs)

            decoder_input = torch.bmm(attn_probs.unsqueeze(1), encoder_outputs)

        return torch.stack(pointers, dim=1)

