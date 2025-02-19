pre_train.py 构建了英语和日语的词典，分别用cbow和skip-gram训练了word embedding
lstm_attn.py 定义采用lstm和attention的rnn模型，并定义了训练函数。
log.txt 记录了训练日志

skipgram/cbow_model_eng/jpn.model 是word2vector模型
skipgram/cbow_embedding_matrix_eng/jpn.pt 是针对词典的word embedding matrix
encoder_cbow.pth 是训练好的encoder
decoder_cbow.pth 是训练好的decoder
