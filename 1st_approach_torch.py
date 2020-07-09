import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import copy
from argparse import Namespace
import es_core_news_lg


# Hyper parameters
flags = Namespace(
    train_file='Source/TXT/text.txt',            # Source file path
    seq_size=32,                                 # Depth of the recurrence
    batch_size=16,                               # training batch size
    embedding_size=64,                           # Dimension of the embedding vector
    lstm_size=64,                                # Size of the lstm layer
    gradients_norm=5,                            # Limits for clamping the gradients
    initial_words=['Yo', 'soy'],                   # First words to start predicting
    predict_top_k=5,                             # Number os most used words to choose between
    checkpoint_path='checkpoint_torch/models/',  # Path for saving trained models
    result_path='checkpoint_torch/result/',      # Path for saving results
    epochs=2000,                                 # Epochs for training
    n_words=100,                                 # Number of words for the prediction
    learning_rate=1e-3,                          # Learning rate
)


# Function for data treatment
def get_data(train_file, batch_size, seq_size):
    # Get the text in a variable and separate it in word tokens
    with open(train_file, 'r') as file:
        text = file.read()
    text.lower()
    # Charging spaCy dictionary
    nlp = es_core_news_lg.load()
    nlp.max_length = 1000000
    doc = nlp(text)
    # We save the embedding vectors and the words list
    vectors = []
    texts = []
    for v in doc:
        if v.is_oov:
            list.append(vectors, v.vector)
            list.append(texts, v.text)

    # For every word in the text we count the number of times repeated
    word_counts = Counter(texts)
    # We sort the words from the most used to the less used
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # Dictionary from 0 to ... with the most used word (int to string)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    # Inverse action from above
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)
    print('Vocabulary size', n_vocab)

    # Transform tokens into integer values
    int_text = [vocab_to_int[w] for w in texts]
    # Number of batches
    num_batches = int(len(int_text) / (seq_size * batch_size))
    # Input for the nn. Adjusting the data to fit in the batch and length size
    in_text = int_text[:num_batches * batch_size * seq_size]
    # Creating the nn target
    out_text = np.zeros_like(in_text)
    # Decale in one the target (first output will be second input and so forth)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    # Reshape in and out for the network, dividing in columns for each batch
    in_text = np.reshape(in_text, (batch_size, -1))    # dim: (batch_size, length(in_text)/batch_size)
    out_text = np.reshape(out_text, (batch_size, -1))  # dim: (batch_size, length(in_text)/batch_size)
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = int(np.prod(in_text.shape) // (seq_size * batch_size))
    # Every time we call this generator it will give us the sequence sized of the batch size
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]  # Dim: (batch_size, seq_size) for x and y


class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        # Defining the net's architecture
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding_layer = nn.Embedding(n_vocab, embedding_size)            # in: n_vocab / out: embedding_size
        self.lstm_layer = nn.LSTM(embedding_size, lstm_size, batch_first=True)  # in: embedding_size / out: lstm_size
        self.dense_layer = nn.Linear(lstm_size, n_vocab)                        # in: lstm_size / out: n_vocab

    def forward_pass(self, x, prev_state):                   # Dim(x): (batch_size, sequence_size)
        embed = self.embedding_layer(x)                      # Dim(embed): (batch_size, sequence_size, embedding_size)
        output, state = self.lstm_layer(embed, prev_state)   # Dim(output): (batch_size, sequence_size, embedding_size)
        logits = self.dense_layer(output)                    # Dim(logits): (batch_size, sequence_size, n_vocab)
        return logits, state

    def initial_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))


def loss_function_train(net, lr=1e-3):
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Function optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return criterion, optimizer


def predict(device, net, words, n_words, vocab_to_int, int_to_vocab, top_k=5):
    # We tell the network we are evaluating
    net.eval()
    # Getting the hidden state and the memory state
    state_h, state_c = net.initial_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    # We pass word by word through the network, updating the memory states
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net.forward_pass(ix, (state_h, state_c))
    # We find the top_k most utilized words from the output and store the indexes
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    # Randomly select a choice from the top_k most used
    choice = np.random.choice(choices[0])  # choice = np.max(choices)
    # We add the word chosen
    words.append(int_to_vocab[choice])
    # We continue to get more words from the choices we select until n_words
    for nothing in range(n_words):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net.forward_pass(ix, (state_h, state_c))
        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])  # choice = np.max(choices)
        words.append(int_to_vocab[choice])
    # Return the text
    output_text = ' '.join(words)
    return output_text


def main():
    # Check if we can work with a GPU or we'll be working with a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Working with {}'.format(device))
    # Data treatment
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data(flags.train_file,
                                                                      flags.batch_size,
                                                                      flags.seq_size)
    # Network creation
    net = RNNModule(n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size)
    # Sending the network to the GPU if possible
    net = net.to(device)
    # Initialise the optimizer and criterion
    criterion, optimizer = loss_function_train(net, flags.learning_rate)
    # Starting the training loop
    iteration = 0
    for e in range(flags.epochs):
        # Create the batches generator
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        # Initialise LSTM states
        state_h, state_c = net.initial_state(flags.batch_size)
        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1
            # We tell torch training mode on
            net.train()
            # Reset all gradients
            optimizer.zero_grad()
            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            # Forward pass execution
            logits, (state_h, state_c) = net.forward_pass(x, (state_h, state_c))
            # Calculate loss value
            loss = criterion(logits.transpose(1, 2), y)
            # Before delivering the loss we need to detach the states
            state_h = state_h.detach()
            state_c = state_c.detach()
            # Get the loss value
            loss_value = loss.item()
            # Perform back-propagation
            loss.backward()
            # Gradient clip for avoiding exploding gradient
            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), flags.gradients_norm)
            # Update the network's parameters
            optimizer.step()

            # Verbose
            if iteration % 10 == 0:
                print('Epoch: {}/{}'.format(e, flags.epochs),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))
            if iteration % 10 == 0:
                initial_words = copy.deepcopy(flags.initial_words)
                out_net = predict(device, net, initial_words, flags.n_words, vocab_to_int, int_to_vocab, top_k=5)
                print(out_net)
                with open(flags.result_path + 'result-{}.txt'.format(iteration), 'w+') as file:
                    file.write(out_net)
                torch.save(net.state_dict(), flags.checkpoint_path + 'model-{}.pth'.format(iteration))


if __name__ == '__main__':
    main()
