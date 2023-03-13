# %%
# Original code by Tae Hwan Jung @graykode
# https://github.com/graykode/nlp-tutorial
# Modifications by @jaspock

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def random_batch():
    random_inputs = []
    random_labels = []
    # batch_size elements randomly taken from the list of indexes without replacement
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        # eye = I (identity matrix, pronounced as "eye")
        # easy way of creating a one-hot vector:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target
        random_labels.append(skip_grams[i][1])  # context word

    return random_inputs, random_labels

""" The original skip-gram paper computes p(w_t | w), that is, a probability distribution
 over words given the context word w_t. The desired output is a one-hot vector with the
 index of the context word set to 1, and the rest of the vector set to 0. Therefore,
 positive and negative information is computed at the same time. The embedding for
 the target word in W and all the embeddings in WT are updated for a single target word."""

# model
class Word2Vec(nn.Module):
    def __init__(self):
        super().__init__() # recommended over super(Word2Vec, self).__init__()
        # W and WT is not transpose relationship in spite of the names
        self.W = nn.Linear(voc_size, embedding_size, bias=False) # voc_size > embedding_size Weight
        self.WT = nn.Linear(embedding_size, voc_size, bias=False) # embedding_size > voc_size Weight

    def forward(self, X):
        # X : [batch_size, voc_size], X is a one-hot vector
        hidden_layer = self.W(X) # hidden_layer : [batch_size, embedding_size]
        output_layer = self.WT(hidden_layer) # output_layer : [batch_size, voc_size]
        return output_layer

if __name__ == '__main__':
    batch_size = 2 # mini-batch size
    embedding_size = 2 # embedding size

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # Make skip gram of one size window
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()

    # CrossEntropyLoss = LogSoftmax + NLLLoss. 
    # The index of the true class is passed as target, not a one-hot vector.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        # creating a tensor from a single numpy.ndarray is much faster than from a list of numpy.ndarrays:
        input_batch = torch.Tensor(np.array(input_batch)) 
        # indexes are usually represented with longs:
        target_batch = torch.LongTensor(target_batch)  # same as torch.tensor(target_batch, dtype=torch.long)

        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        # See the "variables" section in Linear doc: 
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # The learnable weights of the module are of shape (out_features,in_features)
        # As mentioned, the weights A are used transposed (y = x * A^T + b)
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()

# %%
