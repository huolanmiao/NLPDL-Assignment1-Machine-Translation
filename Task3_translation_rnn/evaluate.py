import torch
from torch import nn
from pre_train import SOS_token, EOS_token, MAX_LENGTH, EngLang, JpnLang
from lstm_attn import load_file, load_data, evaluate
import random
from nltk.translate.bleu_score import sentence_bleu
import math
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_jpn, all_eng = load_data('./eng_jpn.txt')
English = EngLang()
Japanese = JpnLang()
English.build_vocab(all_eng)
Japanese.build_vocab(all_jpn)
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights
    
class EncoderLSTM(nn.Module):
    def __init__(self, embedding_matrix, input_size, hidden_size, dropout_p=0.1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        # embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded)
        return output, (hidden, cell)

class AttnDecoderLSTM(nn.Module):
    def __init__(self, embedding_matrix, output_size, embedding_dim, hidden_size, num_layers, dropout = 0.1):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attention = BahdanauAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        # embedded = self.embedding(input)
        query = hidden[0].permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        # print(f"context: {context.shape}")
        # print(f"embedded: {embedded.shape}")
        input_lstm = torch.cat((embedded, context), dim=2)

        output, hidden = self.lstm(input_lstm, hidden)
        output = self.fc(output)

        return output, hidden, attn_weights

    def forward(self, encoder_outputs, hidden, target_tensor = None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = hidden
        decoder_outputs = []
        attentions = []
        # print(f"encoder_outputs: {encoder_outputs.shape}")
        
        for t in range(MAX_LENGTH):
            # print(f"decoder_hidden: {decoder_hidden[0].shape}")
            # print(f"decoder_input: {decoder_input.shape}")
            output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            # print(f"output: {output.shape}")
            decoder_outputs.append(output)
            attentions.append(attn_weights)
            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, t].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, (hidden, attentions)
# Load model
cbow_embedding_matrix_eng = load_file('cbow_embedding_matrix_eng.pt').to(device)
cbow_embedding_matrix_jpn = load_file('cbow_embedding_matrix_jpn.pt').to(device)
skipgram_embedding_matrix_eng = load_file('skipgram_embedding_matrix_eng.pt').to(device)
skipgram_embedding_matrix_jpn = load_file('skipgram_embedding_matrix_jpn.pt').to(device)
hidden_size = 100
encoder_state_dict = torch.load('encoder_cbow.pth')
decoder_state_dict = torch.load('decoder_cbow.pth')

encoder = EncoderLSTM(cbow_embedding_matrix_jpn, Japanese.n_words, hidden_size).to(device)
decoder = AttnDecoderLSTM(cbow_embedding_matrix_eng, English.n_words, 100, hidden_size, 1, 0.1).to(device)
# encoder.load_state_dict(torch.load('./encoder_cbow.pth'), map_location=device)
# decoder.load_state_dict(torch.load('./decoder_cbow.pth'), map_location=device)  
# torch.save(encoder, 'encoder.pth')
# torch.save(decoder, 'decoder.pth')
encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)
print("load model successfully")
encoder.eval()
decoder.eval()
# Load data
test_eng_index = torch.load('test_eng.pt').to(device)
test_jpn_index = torch.load('test_jpn.pt').to(device)
val_eng_index = torch.load('val_eng.pt').to(device)
val_jpn_index = torch.load('val_jpn.pt').to(device)


val_dataloader = TensorDataset(val_jpn_index.to(device), val_eng_index.to(device))
val_dataloader = DataLoader(val_dataloader, batch_size=64, shuffle=True)
test_dataloader = TensorDataset(test_jpn_index.to(device), test_eng_index.to(device))
test_dataloader = DataLoader(test_dataloader, batch_size=64, shuffle=True)

val_loss, val_bleu, val_ppl = evaluate(val_dataloader, encoder, decoder)
test_loss, test_bleu, test_ppl = evaluate(test_dataloader, encoder, decoder)
print(f'Validation loss: {val_loss:.4f}, BLEU score: {val_bleu:.4f}, Perplexity: {val_ppl:.4f}')
print(f'Test loss: {test_loss:.4f}, BLEU score: {test_bleu:.4f}, Perplexity: {test_ppl:.4f}')

# translate the sentence
case_1 = "私の名前は愛です"
case_2 = "昨日はお肉を食べません"
case_3 = "いただきますよう"
case_4 = "秋は好きです"
case_5 = "おはようございます"

def translate(case, encoder, decoder):
    input_sentence = Japanese.tokenize(case)
    indices = [Japanese.word2index[word] for word in input_sentence]
    indices.append(EOS_token)
    input_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden)
    decoded_words = []
    for di in range(MAX_LENGTH):
        topv, topi = decoder_outputs[0, di].topk(1)
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(English.index2word[topi.item()])
    return decoded_words

print(f"Case 1: {case_1} -> {' '.join(translate(case_1, encoder, decoder))}")
print(f"Case 2: {case_2} -> {' '.join(translate(case_2, encoder, decoder))}")
print(f"Case 3: {case_3} -> {' '.join(translate(case_3, encoder, decoder))}")
print(f"Case 4: {case_4} -> {' '.join(translate(case_4, encoder, decoder))}")
print(f"Case 5: {case_5} -> {' '.join(translate(case_5, encoder, decoder))}")


