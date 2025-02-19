import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def get_cosine_similarity(embedding_1, embedding_2):
    return cosine_similarity([embedding_1], [embedding_2])[0][0]

# Load word embedding 

cbow_eng = gensim.models.Word2Vec.load('./cbow_model_eng.model')
word_embedding = cbow_eng.wv
# skipgram_eng = gensim.models.Word2Vec.load('./skipgram_model_eng.model')
# word_embedding = skipgram_eng.wv

# # (随机)选取50个词,对word embedding进行可视化

# words = list(word_embedding.index_to_key)[:50]
# embedding_matrix = np.array([word_embedding[word] for word in words])


# # 使用PCA将高维嵌入向量降维到2D
# pca = PCA(n_components=2)
# embedding_2d = pca.fit_transform(embedding_matrix)

# # 可视化降维后的嵌入
# plt.figure(figsize=(10, 8))
# for i, word in enumerate(words):
#     plt.scatter(embedding_2d[i, 0], embedding_2d[i, 1])
#     plt.text(embedding_2d[i, 0] + 0.02, embedding_2d[i, 1] + 0.02, word, fontsize=12)
# plt.title("2D PCA of Cbow Word Embeddings ")
# plt.show()



# # 使用t-SNE将嵌入向量降维到2D
# tsne = TSNE(n_components=2, random_state=42)
# embedding_2d_tsne = tsne.fit_transform(embedding_matrix)

# # 可视化t-SNE结果
# plt.figure(figsize=(10, 8))
# for i, word in enumerate(words):
#     plt.scatter(embedding_2d_tsne[i, 0], embedding_2d_tsne[i, 1])
#     plt.text(embedding_2d_tsne[i, 0] + 0.02, embedding_2d_tsne[i, 1] + 0.02, word, fontsize=12)
# plt.title("2D t-SNE of Word Embeddings")
# plt.show()


# # 并排展示PCA和t-SNE的结果
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
# ax1.set_title("2D PCA of Cbow Word Embeddings")
# ax2.set_title("2D t-SNE of Cbow Word Embeddings")
# for i, word in enumerate(words):
#     ax1.scatter(embedding_2d[i, 0], embedding_2d[i, 1])
#     ax1.text(embedding_2d[i, 0] + 0.02, embedding_2d[i, 1] + 0.02, word, fontsize=12)
#     ax2.scatter(embedding_2d_tsne[i, 0], embedding_2d_tsne[i, 1])
#     ax2.text(embedding_2d_tsne[i, 0] + 0.02, embedding_2d_tsne[i, 1] + 0.02, word, fontsize=12)
# plt.show()

import pandas as pd
# Load dataset
dataset_path_sim = './wordsim_similarity_goldstandard.txt'
df_sim = pd.read_csv(dataset_path_sim, sep='\s+', header=None, names=['word1', 'word2', 'similarity'])
dataset_path_rel = './wordsim_relatedness_goldstandard.txt'
df_rel = pd.read_csv(dataset_path_rel, sep='\s+', header=None, names=['word1', 'word2', 'relatedness'])

def compute_similarity(word1, word2, embedding_model):
    vec1 = embedding_model[word1]
    vec2 = embedding_model[word2]
    return cosine_similarity([vec1], [vec2])[0][0]

# print("Similarity between (Tom, Mary): ", compute_similarity('Tom', 'Mary', word_embedding))
# print("Similarity between (He, She): ", compute_similarity('He', 'She', word_embedding))
# print("Similarity between (in, on): ", compute_similarity('in', 'on', word_embedding))
# Similarity between (Tom, Mary):  0.7510916
# Similarity between (He, She):  0.9654538
# Similarity between (in, on):  0.8808856

def is_in_vocabulary(word1, word2, embedding_model):
    return word1 in embedding_model and word2 in embedding_model
# Load word embedding model

# Filter out OOV words
df_filtered_sim = df_sim[df_sim.apply(lambda row: is_in_vocabulary(row['word1'], row['word2'], word_embedding), axis=1)]
df_filtered_rel = df_rel[df_rel.apply(lambda row: is_in_vocabulary(row['word1'], row['word2'], word_embedding), axis=1)]

# Compute cosine similarity for each word pair
df_filtered_sim['embedding_similarity'] = df_filtered_sim.apply(
    lambda row: compute_similarity(row['word1'], row['word2'], word_embedding), axis=1)
df_filtered_rel['embedding_similarity'] = df_filtered_rel.apply(
    lambda row: compute_similarity(row['word1'], row['word2'], word_embedding), axis=1)

# Calculate Spearman correlation
spearman_corr_sim, _ = spearmanr(df_filtered_sim['similarity'], df_filtered_sim['embedding_similarity'])
print(f"Spearman Correlation sim: {spearman_corr_sim:.4f}")
spearman_corr_rel, _ = spearmanr(df_filtered_rel['relatedness'], df_filtered_rel['embedding_similarity'])
print(f"Spearman Correlation rel: {spearman_corr_rel:.4f}")
# skipgram 
# window 5, epochs 5
# Spearman Correlation sim: 0.1071
# Spearman Correlation rel: 0.0167
# window 5, epochs 15
# Spearman Correlation sim: 0.2464
# Spearman Correlation rel: 0.1440
# window 10, epochs 15
# Spearman Correlation sim: 0.2706
# Spearman Correlation rel: 0.1644

# cbow
# window 5, epochs 5
# Spearman Correlation sim: 0.1304
# Spearman Correlation rel: 0.0546
# window 5, epochs 15
# Spearman Correlation sim: 0.2689
# Spearman Correlation rel: 0.1511
# window 10, epochs 15
# Spearman Correlation sim: 0.3100
# Spearman Correlation rel: 0.1955