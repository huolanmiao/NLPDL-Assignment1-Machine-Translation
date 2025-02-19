import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm
from pre_train import EngLang, JpnLang, load_data, SOS_token, EOS_token, MAX_LENGTH
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def showPlot(points, tag):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel(tag)
    plt.plot(points[1:])
    # save as pdf
    plt.savefig(tag+'.pdf')
    
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


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

            
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    total_loss = 0
    bleu_score = 0
    perplexity = 0
    num_sentences = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        
        total_loss += loss.item()
        for i in range(decoder_outputs.size(0)):
            decoded_words = []
            for di in range(MAX_LENGTH):
                topv, topi = decoder_outputs[i, di].topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(English.index2word[topi.item()])
            reference = [English.index2word[idx.item()] for idx in target_tensor[i] if idx.item() != EOS_token]
            
            bleu_score += sentence_bleu([reference], decoded_words, smoothing_function=SmoothingFunction().method4)
            perplexity += math.exp(loss.item())
            num_sentences += 1
            
    avg_loss = total_loss / len(dataloader)
    avg_bleu_score = bleu_score / num_sentences
    avg_perplexity = perplexity / num_sentences
    return avg_loss, avg_bleu_score, avg_perplexity

def evaluate(val_dataloader, encoder, decoder, max_length=MAX_LENGTH, criterion = nn.NLLLoss()):
    encoder.eval()
    decoder.eval()
    total_val_loss = 0
    total_bleu_score = 0
    total_perplexity = 0
    num_sentences = 0
    for data in val_dataloader:
        input_tensor, target_tensor = data
        with torch.no_grad():
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            total_val_loss += loss.item()
            for i in range(decoder_outputs.size(0)):
                decoded_words = []
                for di in range(MAX_LENGTH):
                    topv, topi = decoder_outputs[i, di].topk(1)
                    if topi.item() == EOS_token:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(English.index2word[topi.item()])
                reference = [English.index2word[idx.item()] for idx in target_tensor[i] if idx.item() != EOS_token]
                
                total_bleu_score += sentence_bleu([reference], decoded_words, smoothing_function=SmoothingFunction().method4)
                total_perplexity += math.exp(loss.item())
                num_sentences += 1
                
    avg_loss = total_val_loss / len(val_dataloader)
    avg_bleu_score = total_bleu_score / num_sentences 
    avg_perplexity = total_perplexity / num_sentences 

    return avg_loss, avg_bleu_score, avg_perplexity

def translate(case, encoder, decoder):
    input_sentence = Japanese.tokenize(case)
    sentence_ids = np.zeros((1, MAX_LENGTH), dtype=np.int32)
    indices = [Japanese.word2index[word] for word in input_sentence]
    indices.append(EOS_token)
    sentence_ids[0, :min(MAX_LENGTH, len(indices))] = indices[:min(MAX_LENGTH, len(indices))]
    input_tensor = torch.LongTensor(sentence_ids).to(device)
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

def train(train_dataloader, val_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=1, plot_every=1, test_dataloader=None):
    start = time.time()
    plot_train_losses = []
    plot_val_losses = []
    plot_train_bleu_scores = []
    plot_val_bleu_scores = []
    plot_train_perplexities = []
    plot_val_perplexities = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    epoch_no_improve = 0
    best_dev_loss = float('inf')
    
    # translate the sentence
    case_1 = "私の名前は愛です"
    case_2 = "昨日はお肉を食べません"
    case_3 = "いただきますよう"
    case_4 = "秋は好きです"
    case_5 = "おはようございます"
    for epoch in tqdm(range(1, n_epochs + 1)):
        loss ,bleu, ppl= train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        if epoch % 10 == 0:
            torch.save(encoder, 'encoder_cbow.pth')
            torch.save(decoder, 'decoder_cbow.pth')
            with open('log.txt', 'a') as f:
                f.write(f"Case 1: {case_1} -> {' '.join(translate(case_1, encoder, decoder))}")
                f.write(f"Case 2: {case_2} -> {' '.join(translate(case_2, encoder, decoder))}")
                f.write(f"Case 3: {case_3} -> {' '.join(translate(case_3, encoder, decoder))}")
                f.write(f"Case 4: {case_4} -> {' '.join(translate(case_4, encoder, decoder))}")
                f.write(f"Case 5: {case_5} -> {' '.join(translate(case_5, encoder, decoder))}")
                

        val_loss, val_bleu, val_ppl = evaluate(val_dataloader, encoder, decoder)
        
        
        if epoch % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, loss))
            print(f'Train BLEU: {bleu}, Train PPL: {ppl}')
            print(f'Validation BLEU: {val_bleu}, Validation PPL: {val_ppl}')
            # log in a file
            with open('log.txt', 'a') as f:
                f.write(f'Epoch: {epoch}, Train Loss: {loss}, Train BLEU: {bleu}, Train PPL: {ppl}\n')
                f.write(f'Epoch: {epoch}, Validation Loss: {val_loss}, Validation BLEU: {val_bleu}, Validation PPL: {val_ppl}\n')

        if epoch % plot_every == 0:
            plot_train_losses.append(loss)
            plot_val_losses.append(val_loss)
            plot_train_bleu_scores.append(bleu)
            plot_val_bleu_scores.append(val_bleu)
            plot_train_perplexities.append(ppl)
            plot_val_perplexities.append(val_ppl)
        
        if val_loss < best_dev_loss:
            best_dev_loss = val_loss
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
        
        # Early stopping
        if epoch_no_improve >= 10:
            print('Early stopping triggered')
            torch.save(encoder, 'encoder_cbow.pth')
            torch.save(decoder, 'decoder_cbow.pth')
            break
                

    showPlot(plot_train_losses,'train_losses')
    showPlot(plot_val_losses,'val_losses')
    showPlot(plot_train_bleu_scores,'train_bleu_scores')
    showPlot(plot_val_bleu_scores,'val_bleu_scores')
    showPlot(plot_train_perplexities,'train_perplexities')
    showPlot(plot_val_perplexities,'val_perplexities')
    
    test_loss, test_bleu, test_ppl = evaluate(test_dataloader, encoder, decoder)
    with open('log.txt', 'a') as f:
        f.write(f'Test Loss: {test_loss}, Test BLEU: {test_bleu}, Test PPL: {test_ppl}\n')
        f.write(f"Case 1: {case_1} -> {' '.join(translate(case_1, encoder, decoder))}")
        f.write(f"Case 2: {case_2} -> {' '.join(translate(case_2, encoder, decoder))}")
        f.write(f"Case 3: {case_3} -> {' '.join(translate(case_3, encoder, decoder))}")
        f.write(f"Case 4: {case_4} -> {' '.join(translate(case_4, encoder, decoder))}")
        f.write(f"Case 5: {case_5} -> {' '.join(translate(case_5, encoder, decoder))}")
    
    
    
    

all_jpn, all_eng = load_data('./eng_jpn.txt')
English = EngLang()
Japanese = JpnLang()
English.build_vocab(all_eng)
Japanese.build_vocab(all_jpn)

# load index
def load_file(file_path):
    return torch.load(file_path)

train_eng_index = load_file('train_eng.pt')
train_jpn_index = load_file('train_jpn.pt')
test_eng_index = load_file('test_eng.pt')
test_jpn_index = load_file('test_jpn.pt')
val_eng_index = load_file('val_eng.pt')
val_jpn_index = load_file('val_jpn.pt')

# load embedding matrix
cbow_embedding_matrix_eng = load_file('cbow_embedding_matrix_eng.pt')
cbow_embedding_matrix_jpn = load_file('cbow_embedding_matrix_jpn.pt')
skipgram_embedding_matrix_eng = load_file('skipgram_embedding_matrix_eng.pt')
skipgram_embedding_matrix_jpn = load_file('skipgram_embedding_matrix_jpn.pt')

# cbow
hidden_size = 100
encoder = EncoderLSTM(cbow_embedding_matrix_jpn, Japanese.n_words, hidden_size).to(device)
decoder = AttnDecoderLSTM(cbow_embedding_matrix_eng, English.n_words, 100, hidden_size, 1, 0.1).to(device)

train_dataloader = TensorDataset(train_jpn_index.to(device), train_eng_index.to(device))
train_dataloader = DataLoader(train_dataloader, batch_size=64, shuffle=True)

# 引入validation set
val_dataloader = TensorDataset(val_jpn_index.to(device), val_eng_index.to(device))
val_dataloader = DataLoader(val_dataloader, batch_size=64, shuffle=True)
test_dataloader = TensorDataset(test_jpn_index.to(device), test_eng_index.to(device))
test_dataloader = DataLoader(test_dataloader, batch_size=64, shuffle=True)

train(train_dataloader, val_dataloader, encoder, decoder, 100, learning_rate=0.001, print_every=1, plot_every=1, test_dataloader=test_dataloader)


