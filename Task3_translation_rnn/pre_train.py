import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sudachipy import tokenizer
from sudachipy import dictionary
import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30

class EngLang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def tokenize(self, sentence):
        return sentence.split(' ')
    
    def addSentence(self, sentence):
        # 如何处理标点符号？
        for word in self.tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def build_vocab(self, all_eng):
        # tokenize the eng text
        for sentence in all_eng:
            self.addSentence(sentence)

class JpnLang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        # self.tagger = MeCab.Tagger("-Owakati")
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C


    def addSentence(self, sentence):
        for word in self.tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def tokenize(self, sentence):
        tokens = self.tokenizer_obj.tokenize(sentence, self.mode)
        return [token.surface() for token in tokens]
    
    def build_vocab(self, all_jpn):
        for sentence in all_jpn:
            self.addSentence(sentence)
            
# Load your dataset
def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['jpn', 'eng'])
    return data['jpn'].tolist(), data['eng'].tolist()

all_jpn, all_eng = load_data('./eng_jpn.txt')
English = EngLang()
Japanese = JpnLang()
# Build vocabulary
English.build_vocab(all_eng)
Japanese.build_vocab(all_jpn)

# Use CBOW and Skip–Gram to train the word vectors
# Prepare data for Word2Vec
eng_sentences = [sentence.split(' ') for sentence in all_eng]
jpn_sentences = [Japanese.tokenize(sentence) for sentence in all_jpn]
# Train CBOW model
cbow_model_eng = Word2Vec(sentences=eng_sentences, vector_size=100, window=10, min_count=1, sg=0, epochs = 15)
cbow_model_jpn = Word2Vec(sentences=jpn_sentences, vector_size=100, window=10, min_count=1, sg=0, epochs = 15)
# Train Skip-Gram model
skipgram_model_eng = Word2Vec(sentences=eng_sentences, vector_size=100, window=10, min_count=1, sg=1, epochs = 15)
skipgram_model_jpn = Word2Vec(sentences=jpn_sentences, vector_size=100, window=10, min_count=1, sg=1, epochs = 15)


# Get the embedding matrix
def get_embedding_matrix(model, lang):
    embedding_matrix = torch.zeros(lang.n_words, 100)
    for word, idx in lang.word2index.items():
        if word in model.wv:
            embedding_matrix[idx] = torch.tensor(model.wv[word])
    return embedding_matrix

cbow_embedding_matrix_eng = get_embedding_matrix(cbow_model_eng, English)
cbow_embedding_matrix_jpn = get_embedding_matrix(cbow_model_jpn, Japanese)
skipgram_embedding_matrix_eng = get_embedding_matrix(skipgram_model_eng, English)
skipgram_embedding_matrix_jpn = get_embedding_matrix(skipgram_model_jpn, Japanese)

# Save the Word2Vector models
cbow_model_eng.save("cbow_model_eng.model")
cbow_model_jpn.save("cbow_model_jpn.model")
skipgram_model_eng.save("skipgram_model_eng.model")
skipgram_model_jpn.save("skipgram_model_jpn.model")

# Save embedding matrix
torch.save(cbow_embedding_matrix_eng, "cbow_embedding_matrix_eng.pt")
torch.save(cbow_embedding_matrix_jpn, "cbow_embedding_matrix_jpn.pt")
torch.save(skipgram_embedding_matrix_eng, "skipgram_embedding_matrix_eng.pt")
torch.save(skipgram_embedding_matrix_jpn, "skipgram_embedding_matrix_jpn.pt")

# 到此处退出程序
quit

# Randomly split the corpus into training set, validation set, and test set (the proportion = 8:1:1)
# Combine Japanese and English sentences into pairs
sentence_pairs = list(zip(jpn_sentences, eng_sentences))

# Split the data into training (80%), validation (10%), and test (10%) sets
train_pairs, test_pairs = train_test_split(sentence_pairs, test_size=0.2, random_state=8)
val_pairs, test_pairs = train_test_split(test_pairs, test_size=0.5, random_state=8)

# Separate the pairs back into individual lists
train_jpn, train_eng = zip(*train_pairs)
val_jpn, val_eng = zip(*val_pairs)
test_jpn, test_eng = zip(*test_pairs)

# Save the splits to files
def save_split(file_path, jpn_sentences, eng_sentences):
    with open(file_path, 'w', encoding='utf-8') as f:
        for jpn, eng in zip(jpn_sentences, eng_sentences):
            f.write(f"{jpn}\t{eng}\n")

save_split('train_split.txt', train_jpn, train_eng)
save_split('val_split.txt', val_jpn, val_eng)
save_split('test_split.txt', test_jpn, test_eng)

# Convert the tokenized sentences to indices
def split2index(lang, sentences):
    sentence_ids = np.zeros((len(sentences), MAX_LENGTH), dtype=np.int32)
    for idx, sentence in enumerate(sentences):
        indices = [lang.word2index[word] for word in sentence]
        indices.append(EOS_token)
        sentence_ids[idx, :min(MAX_LENGTH, len(indices))] = indices[:min(MAX_LENGTH, len(indices))]
    return torch.LongTensor(sentence_ids)

train_eng = split2index(English, train_eng)
val_eng = split2index(English, val_eng)
test_eng = split2index(English, test_eng)
train_jpn = split2index(Japanese, train_jpn)
val_jpn = split2index(Japanese, val_jpn)
test_jpn = split2index(Japanese, test_jpn)

# Save the indices 
torch.save(train_eng, 'train_eng.pt')
torch.save(val_eng, 'val_eng.pt')
torch.save(test_eng, 'test_eng.pt')
torch.save(train_jpn, 'train_jpn.pt')
torch.save(val_jpn, 'val_jpn.pt')
torch.save(test_jpn, 'test_jpn.pt')

