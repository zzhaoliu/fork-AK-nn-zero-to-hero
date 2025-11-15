#implementation of makemore part2 Multi-layer perceptron on a new dataset

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# build the chinese character vocabulary dictionary
# lst = open('./Ancient_Names_Corpus（25W）.txt', 'r').read().splitlines()[4:] # The dataset is from https://github.com/wainshine/Chinese-Names-Corpus/blob/master/Chinese_Names_Corpus/Ancient_Names_Corpus%EF%BC%8825W%EF%BC%89.txt
lst = open('./Chinese_Names_Corpus（120W）.txt', 'r').read().splitlines()[4:] # The dataset is from https://github.com/wainshine/Chinese-Names-Corpus/blob/master/Chinese_Names_Corpus/Chinese_Names_Corpus%EF%BC%88120W%EF%BC%89.txt
words = sorted(list(set(''.join(lst))))
words = ['.'] + words
w2i = {s:i for i, s in enumerate(words)}
# i2w = {i:s for i, s in enumerate(words)}
i2w = {i:s for s,i in w2i.items()}

# build the dataset (a 3 gram dataset) 
# No. gram can be tuned, it's a hyperparameter on the model side, more gram -> more char considered for each inference == increase context window
gram = 6
X = []; Y = []
for s in lst:
    s = s + '.'
    context = [0] * gram
    for w in s:
        # print(''.join(i2w[i] for i in context), '--->', w)
        X.append(context)
        Y.append(w2i[w])
        context = context[1:] + [w2i[w]]
X = torch.tensor(X); Y = torch.tensor(Y); print(f'input shape: {X.shape}\ninput dtype: {X.dtype}\noutput shape: {Y.shape}\noutput dtype: {Y.dtype}\n')

# split the dataset into train, validation and test sets
X_train = X[:int(0.8*len(X))]; Y_train = Y[:int(0.8*len(Y))]; print(f'train set size: X:{X_train.shape}, Y:{Y_train.shape}')
X_val = X[int(0.8*len(X)):int(0.9*len(X))]; Y_val = Y[int(0.8*len(Y)):int(0.9*len(Y))]; print(f'valid set size: X:{X_val.shape}, Y:{Y_val.shape}')
X_test = X[int(0.9*len(X)):]; Y_test = Y[int(0.9*len(Y)):]; print(f'test set size: X:{X_test.shape}, Y:{Y_test.shape}\n')

# build the embedding layer
g = torch.Generator().manual_seed(123456) # for reproducibility
word_vec_dim = 2
W = torch.randn((len(words), word_vec_dim), generator=g)

# build the first hidden layer (weight dim = (gram * word_vec_dim, num neurons in this layer))
layer_1_dim = 100
W1 = torch.randn((gram * word_vec_dim, layer_1_dim), generator=g)
b1 = torch.randn(layer_1_dim, generator=g)

# build the second hidden layer (weight dim = (layer_1_dim, categories in the output))
layer_2_dim = len(words)
W2 = torch.randn((layer_1_dim, layer_2_dim), generator=g)
b2 = torch.randn(layer_2_dim, generator=g)

# indicate to pytorch to track the gradients of the parameters
parameters = [W, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
print(f'Number of parameters in total: {sum(p.nelement() for p in parameters)}')

# training loop
train_iterations = 500
lossi = []
iteri = []
batch_size = 32
lr = 0.1
for _ in range(train_iterations):
    # minibatch construct
    ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)

    # forward pass
    emb = W[X_train[ix]].view(-1, gram * word_vec_dim)
    h1_out = torch.tanh(emb @ W1 + b1)
    h2_out = h1_out @ W2 + b2 # h2_out is also the logits
    loss = F.cross_entropy(h2_out, Y_train[ix])

    # backward pass
    for p in parameters:
        p.grad = None # clear the gradients values
    loss.backward() # compute gradients
    
    # update parameters
    for p in parameters:
        p.data += -lr * p.grad
    
    # track stats
    iteri.append(_)
    lossi.append(loss.log10().item()) # log10 makes the visualisation of loss change more linear, more intuitive to inspect

# visualise the loss against iteration curve on train set
plt.plot(iteri, lossi)
plt.xlabel('iteration')
plt.ylabel('log10(cross entropy loss)')
plt.show()

# evaluate the model on train and validation sets
total_loss = 0
num_of_batches = 100
for i in range(num_of_batches):
    len_of_minibatch = int(len(X_train) / num_of_batches)
    emb = W[X_train[i*len_of_minibatch:(i+1)*len_of_minibatch]].view(-1, gram * word_vec_dim)
    h1_out = torch.tanh(emb @ W1 + b1)
    h2_out = h1_out @ W2 + b2
    loss = F.cross_entropy(h2_out, Y_train[i*len_of_minibatch:(i+1)*len_of_minibatch])
    # if i % 10 == 0:
    #     print(f'loss on batch {i}/{num_of_batches} of train set: {loss.item()}')
    total_loss += loss.item()
mean_loss = total_loss / num_of_batches
print(f'loss on train set: {mean_loss}')

for i in range(num_of_batches):
    len_of_minibatch = int(len(X_val) / num_of_batches)
    emb = W[X_val[i*len_of_minibatch:(i+1)*len_of_minibatch]].view(-1, gram * word_vec_dim)
    h1_out = torch.tanh(emb @ W1 + b1)
    h2_out = h1_out @ W2 + b2
    loss = F.cross_entropy(h2_out, Y_val[i*len_of_minibatch:(i+1)*len_of_minibatch])
    # if i % 10 == 0:
    #     print(f'loss on batch {i}/{num_of_batches} of validation set: {loss.item()}')
    total_loss += loss.item()
mean_loss = total_loss / num_of_batches
print(f'loss on validation set: {mean_loss}')

# evaluate the model on test set for final evaluation
for i in range(num_of_batches):
    len_of_minibatch = int(len(X_test) / num_of_batches)
    emb = W[X_test[i*len_of_minibatch:(i+1)*len_of_minibatch]].view(-1, gram * word_vec_dim)
    h1_out = torch.tanh(emb @ W1 + b1)
    h2_out = h1_out @ W2 + b2
    loss = F.cross_entropy(h2_out, Y_test[i*len_of_minibatch:(i+1)*len_of_minibatch])
    # if i % 10 == 0:
    #     print(f'loss on batch {i}/{num_of_batches} of test set: {loss.item()}')
    total_loss += loss.item()
mean_loss = total_loss / num_of_batches
print(f'loss on test set: {mean_loss}')

# generate text from the model
num_of_generation = 20
print(f'\nGenerating {num_of_generation} names:')
for _ in range(num_of_generation):
    context = [0] * gram
    name = []
    while True:
        emb = W[torch.tensor(context)]
        h1_out = torch.tanh(emb.view(-1, gram * word_vec_dim) @ W1 + b1)
        h2_out = h1_out @ W2 + b2
        probs = F.softmax(h2_out, dim=1)
        i = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [i]
        if i == 0:
            break
        name.append(i2w[i])
    name = ''.join(name)
    print(f'{_}th: {name}')

# I feel like the part of video on the diagnosis of the performance bottleneck is a gem,
# the thought process of diagnosing the performance bottleneck is very valuable,
# for making decisions in practical development that improves the performance efficiently,
# maybe I should watch that part again and think about it more deeply.
