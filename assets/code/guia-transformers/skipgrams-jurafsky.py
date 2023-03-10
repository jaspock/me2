# %%
# Original code by Tae Hwan Jung @graykode
# https://github.com/graykode/nlp-tutorial
# Modifications by @jaspock

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import itertools
import random

def get_3word_sentences():
  words = [
      ["monkey", "cat", "dog", "animal"],
      ["apple", "banana", "fruit", "orange"],
      ["pencil", "pen", "notebook", "book"],
      ["car", "bicycle", "plane", "ship"],
      ["red", "green", "blue", "yellow"],
      ["flute", "violin", "guitar", "drum"]
  ]

  combinations = []
  for group in words:
      for combination in itertools.permutations(group, 3):
          combinations.append(" ".join(combination))
  return combinations


def random_batch():
    random_target = []
    random_context= []
    random_output = []

    # batch_size elements randomly taken from the list of indexes without replacement
    random_index = np.random.choice(range(len(skip_grams_positive)), base_batch_size, replace=False)
    for i in random_index:
        random_target.append(skip_grams_positive[i][0])
        random_context.append(skip_grams_positive[i][1])
        random_output.append(1)

    random_index = np.random.choice(range(len(skip_grams_negative)), base_batch_size*k, replace=False)
    for i in random_index:
        random_target.append(skip_grams_negative[i][0])
        random_context.append(skip_grams_negative[i][1])
        random_output.append(0)   

    return random_target, random_context, random_output


# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Embedding(voc_size, embedding_size)  # target matrix 
        self.C = nn.Embedding(voc_size, embedding_size)  # context matrix

    def forward(self, Xt, Xc):
        # Xt, Xc : [batch_size,1]
        w = self.W(Xt)  # [batch_size, embedding_size]
        c = self.C(Xc)  # [batch_size, embedding_size]
        dot = torch.einsum('ij,ji->i',w,c.t())  # efficient computation of all necessary scalar products
        output_layer = torch.sigmoid(dot)
        return output_layer

if __name__ == '__main__':

    base_batch_size = 256 # mini-batch size will be base_batch_size*(k+1)
    embedding_size = 2 # embedding size
    k = 2 # ratio negative/positive samples

    sentences = get_3word_sentences()

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()  
    word_list = list(set(word_list))  # removes duplicates
    word_dict = {w: i for i, w in enumerate(word_list)}  # word from index
    dict_word = {i: w for i, w in enumerate(word_list)}  # index from word
    voc_size = len(word_list)

    # Make skip gram of one size window
    skip_grams_positive = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for c in context:
            skip_grams_positive.append([target, c])

    # Generate roughly k negative samples for each positive
    skip_grams_negative = []
    for i in range(len(word_sequence)):
        target = word_dict[word_sequence[i]]
        context = [word_dict[i] for i in random.sample(word_list,k)]
        for c in context:
            skip_grams_negative.append([target, c])

    # Print samples
    print('Some positive samples: ', end='')
    for i in skip_grams_positive[::len(skip_grams_positive)//20]:
      print(dict_word[i[0]], dict_word[i[1]],end=', ')
    print('\nSome negative samples: ', end='')
    for i in skip_grams_negative[::len(skip_grams_negative)//20]:
      print(dict_word[i[0]], dict_word[i[1]],end=', ')
    print()

    model = Word2Vec()

    criterion = nn.BCELoss()
    # model.parameters(): iterable with all parameters of the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(10000):
        input_w_batch, input_c_batch, target_batch = random_batch()
        # indexes are usually represented with longs in PyTorch:
        input_w_batch = torch.tensor(input_w_batch,dtype=torch.long)
        input_c_batch = torch.tensor(input_c_batch,dtype=torch.long)
        target_batch = torch.tensor(target_batch,dtype=torch.float32) # BCELoss expects float values

        optimizer.zero_grad()
        output = model(input_w_batch,input_c_batch)

        # output : [batch_size], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch: {(epoch+1):04d}, cost={loss:.6f}')

        loss.backward()
        optimizer.step()

    plt.figure(figsize=(8, 8))
    for i, label in enumerate(word_list):
        W, C = model.parameters()
        x, y = W[i][0].item(), W[i][1].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
