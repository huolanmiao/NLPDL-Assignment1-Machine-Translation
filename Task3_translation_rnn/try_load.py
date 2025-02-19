import torch
from torch import nn
from lstm_attn import load_file
# Define the model architecture (ensure it matches the saved model)
class EncoderLSTM(nn.Module):
    def __init__(self, embedding_matrix, input_size, hidden_size, dropout_p=0.1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded)
        return output, (hidden, cell)

class AttnDecoderLSTM(nn.Module):
    def __init__(self, embedding_matrix, output_size, embedding_dim, hidden_size, num_layers, dropout=0.1):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attention = BahdanauAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden[0].permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_lstm = torch.cat((embedded, context), dim=2)
        output, hidden = self.lstm(input_lstm, hidden)
        output = self.fc(output)
        return output, hidden, attn_weights

    def forward(self, encoder_outputs, hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = hidden
        decoder_outputs = []
        attentions = []
        for t in range(MAX_LENGTH):
            output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(output)
            attentions.append(attn_weights)
            if target_tensor is not None:
                decoder_input = target_tensor[:, t].unsqueeze(1)  # Teacher forcing
            else:
                _, topi = output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, (hidden, attentions)

# Load the state dictionary
encoder_state_dict = torch.load('encoder_cbow.pth')
decoder_state_dict = torch.load('decoder_cbow.pth')

# Initialize the model
hidden_size = 100
cbow_embedding_matrix_eng = load_file('cbow_embedding_matrix_eng.pt').to(device)
cbow_embedding_matrix_jpn = load_file('cbow_embedding_matrix_jpn.pt').to(device)
encoder = EncoderLSTM(cbow_embedding_matrix_jpn, Japanese.n_words, hidden_size).to(device)
decoder = AttnDecoderLSTM(cbow_embedding_matrix_eng, English.n_words, 100, hidden_size, 1, 0.1).to(device)

# Load the state dictionary into the model
encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)

# Set the model to evaluation mode
encoder.eval()
decoder.eval()

print("Model loaded successfully")