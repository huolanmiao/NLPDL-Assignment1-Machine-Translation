import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import pandas as pd
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tokenizer(text):
    # Use jieba to tokenize Chinese text
    return [word for word in jieba.cut(text)]

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
    return data['text'].tolist(), data['label'].tolist()

# 1.Load the data
train_texts, train_labels = load_data('./train.txt')
dev_texts, dev_labels = load_data('./dev.txt')
test_texts, test_labels = load_data('./test.txt')

# 2.Tokenize the texts
train_tokens = [tokenizer(text) for text in train_texts]
dev_tokens = [tokenizer(text) for text in dev_texts]
test_tokens = [tokenizer(text) for text in test_texts]

# Get max length of the sentences. 
max_len = max(len(tokens) for tokens in train_tokens)
print(f'Maximum sentence length: {max_len}')
#The maximum sentence length is 20
max_len = 20  

# 3.Build vocabulary
vocab = Counter(token for tokens in train_tokens for token in tokens)
vocab = {word: idx for idx, (word, _) in enumerate(vocab.items(), start=2)}
# Add special tokens
vocab['<pad>'] = 0
vocab['<unk>'] = 1

# 4.Convert tokens to indices
def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

train_indices = [tokens_to_indices(tokens, vocab) for tokens in train_tokens]
dev_indices = [tokens_to_indices(tokens, vocab) for tokens in dev_tokens]
test_indices = [tokens_to_indices(tokens, vocab) for tokens in test_tokens]

# Pad sequences
def pad_sequences(sequences, max_len, pad_value=0):
    return [seq[:max_len] + [pad_value] * max(0, max_len - len(seq)) for seq in sequences]

train_indices = pad_sequences(train_indices, max_len)
dev_indices = pad_sequences(dev_indices, max_len)
test_indices = pad_sequences(test_indices, max_len)

# 5.Convert to PyTorch tensors
train_data = torch.utils.data.TensorDataset(torch.tensor(train_indices).to(device), torch.tensor(train_labels).to(device))
dev_data = torch.utils.data.TensorDataset(torch.tensor(dev_indices).to(device), torch.tensor(dev_labels).to(device))
test_data = torch.utils.data.TensorDataset(torch.tensor(test_indices).to(device), torch.tensor(test_labels).to(device))

# Create data loaders
batch_size = 64
train_iterator = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
dev_iterator = torch.utils.data.DataLoader(dev_data, batch_size=batch_size)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


# 6. Define the CNN-based model for sentence classification
class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout=0.5):
        super(CNN, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Convolution layers with different filter sizes (for different n-grams)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embed_dim))
            for fs in filter_sizes
        ])

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: [batch_size, sentence_length]
        embedded = self.embedding(x)  # [batch_size, sentence_length, embed_dim]
        embedded = embedded.unsqueeze(1)  # Add a channel dimension: [batch_size, 1, sentence_length, embed_dim]

        # Convolution & Max pooling
        conv_results = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled_results = [torch.max(result, dim=2)[0] for result in conv_results]

        # Concatenate the pooled features
        cat = torch.cat(pooled_results, dim=1)

        # Apply dropout
        dropped = self.dropout(cat)

        # Fully connected layer
        return self.fc(dropped)

# 7. Hyperparameters and model initialization
vocab_size = len(vocab)
embed_dim = 100
num_classes = 4
filter_sizes = [3, 4, 5]  # Common n-gram sizes (trigrams, 4-grams, 5-grams)
num_filters = 10
dropout = 0.5

model = CNN(vocab_size, embed_dim, num_classes, filter_sizes, num_filters, dropout)

# 8. Define criterion and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 9. Train and evaluate
model = model.to(device)
criterion = criterion.to(device)

def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch[0]).squeeze(1)
        loss = criterion(predictions, batch[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        correct = (torch.argmax(predictions, dim=1) == batch[1]).float()
        epoch_correct += correct.sum().item()
    return epoch_loss / len(iterator), epoch_correct/(64*len(iterator))

def evaluate_model(model, iterator, criterion):
    # Calculate loss and acc
    model.eval()
    epoch_loss = 0
    epoch_acc = 0   
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch[0]).squeeze(1)
            loss = criterion(predictions, batch[1])
            epoch_loss += loss.item()
            acc = (torch.argmax(predictions, dim=1) == batch[1]).float()
            epoch_acc += acc.sum().item()
            
    return epoch_loss / len(iterator), epoch_acc / (64*len(iterator))

patience = 5
best_dev_loss = float('inf')
epochs_no_improve = 0

log_train_loss ,log_train_acc= [], []
log_dev_loss ,log_dev_acc = [], []
# Try changing max_len, num_filters, embed_dim, patience
for epoch in range(50):
    train_loss ,train_acc= train_model(model, train_iterator, optimizer, criterion)
    dev_loss ,dev_acc = evaluate_model(model, dev_iterator, criterion)
    
    log_train_loss.append(train_loss)
    log_train_acc.append(train_acc)
    log_dev_loss.append(dev_loss)
    log_dev_acc.append(dev_acc)
    print(f"Epoch: {epoch}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Dev loss: {dev_loss:.4f}, Dev acc: {dev_acc:.4f}")
    
    # Check for improvement
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    
    # Early stopping
    if epochs_no_improve >= patience:
        print('Early stopping triggered')
        break

test_loss , test_acc = evaluate_model(model, test_iterator, criterion)
print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
# Save the model
torch.save(model.state_dict(), 'model.pt')

# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(log_train_loss, label='train')
plt.plot(log_dev_loss, label='dev')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(log_train_acc, label='train')
plt.plot(log_dev_acc, label='dev')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
