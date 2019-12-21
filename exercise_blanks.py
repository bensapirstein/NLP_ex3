import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 10
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    vecs = np.array([word_to_vec.get(word, np.zeros(embedding_dim)) for word in sent.text])
    return np.average(vecs, axis=0).astype(np.double)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    res = np.zeros(size)
    res[ind] = 1
    return res


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    one_hots = np.array([get_one_hot(len(word_to_ind), word_to_ind[word]) for word in sent.text])
    return np.average(one_hots, axis=0)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: i for i, word in enumerate(set(words_list))}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    vecs = np.array([word_to_vec.get(word, np.zeros(embedding_dim)) for word in sent.text])
    if len(vecs) >= seq_len:
        return vecs[:seq_len].astype(np.double)
    return np.concatenate((vecs, np.zeros((seq_len - len(vecs), embedding_dim)))).astype(np.double)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list, True),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list, True),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_size = 64
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.hidden2sent = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        # text is of shape (seq_len=10,batch_size,input_size)
        lstm_out, _ = self.lstm(text)
        return self.hidden2sent(lstm_out[:, -1, :]).squeeze()

    def predict(self, text):
        return self.sigmoid(self.forward(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.linear(x).squeeze()

    def predict(self, x):
        return self.sigmoid(self.forward(x))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return np.count_nonzero(preds.round() == y) / len(y)



def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    losses = []
    accs = []
    for i, (x_tensor, y_tensor) in enumerate(data_iterator):
        optimizer.zero_grad()
        y = model(x_tensor.float())
        loss = criterion(y, y_tensor.float())
        loss.backward()
        pred = model.predict(x_tensor.float())
        acc = binary_accuracy(pred, y_tensor.float())
        losses.append(loss.item())
        accs.append(acc)
        optimizer.step()

        if i % 100 == 0:
            print("loss at step %d: %.4f" % (i + 1, loss.item()))
    return np.average(accs), np.average(losses)


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    losses = []
    accs = []
    for i, (x_tensor, y_tensor) in enumerate(data_iterator):
        y = model(x_tensor.float())
        loss = criterion(y, y_tensor.float())
        pred = model.predict(x_tensor.float())
        acc = binary_accuracy(pred, y_tensor.float())
        losses.append(loss.item())
        accs.append(acc)
    return np.average(accs), np.average(losses)


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    return


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = data_manager.get_torch_iterator()
    val_loader = data_manager.get_torch_iterator(VAL)
    test_loader = data_manager.get_torch_iterator(TEST)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(n_epochs):
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_loss = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print("Epoch %d: Train Loss: %.3f Train Acc: %.2f Val Loss: %.3f Val Acc: %.2f" %
              (epoch + 1, train_loss, train_acc, val_loss, val_acc))

    test_acc, test_loss = evaluate(model, test_loader, criterion)
    print("Test Loss: %.3f Test Acc: %.2f" % (test_loss, test_acc))

    NP_acc, RW_acc = special_cases_acc(data_manager, model)
    print("Negated Polarity Acc: %d Rare Words Acc: %d" % (NP_acc, RW_acc))
    return train_losses, train_accs, val_losses, val_accs, test_loss, test_acc, NP_acc, RW_acc


def special_cases_acc(data_manager, model):
    test_sents = data_manager.sentences[TEST]
    test_batches = list(data_manager.get_torch_iterator(TEST))
    test_vectors = []
    for batch in test_batches:
        for vector in batch[0]:
            test_vectors.append(vector)
    test_labels = data_manager.get_labels(TEST)
    NP_sents_idxs = data_loader.get_negated_polarity_examples(test_sents)
    NP_x = [test_vectors[i] for i in NP_sents_idxs]
    NP_y = [test_labels[i] for i in NP_sents_idxs]
    pred = np.array([model.predict(NP_x[i].float()).detach().numpy() for i in range(len(NP_x))])
    NP_acc = binary_accuracy(pred, NP_y)

    RW_sents_idxs = data_loader.get_rare_words_examples(test_sents, data_manager.sentiment_dataset)
    RW_x = [test_vectors[i] for i in RW_sents_idxs]
    RW_y = [test_labels[i] for i in RW_sents_idxs]
    pred = np.array([model.predict(RW_x[i].float()).detach().numpy() for i in range(len(RW_x))])
    RW_acc = binary_accuracy(pred, RW_y)

    return NP_acc, RW_acc


def plots(title, train_losses, train_accs, val_losses, val_accs):
    plt.plot(range(1, len(train_accs) + 1), train_accs)
    plt.plot(range(1, len(val_accs) + 1), val_accs)
    plt.title(title + " - Accuracy")
    plt.ylim((0, 1))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    print("Saving figure at " + title + " - Accuracy.png...")
    plt.savefig(title + " - Accuracy.png")
    plt.clf()
    # summarize history for loss
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.plot(range(1, len(val_losses) + 1), val_losses)
    plt.title(title + " - Loss")
    plt.ylim((0, 1))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    print("Saving figure at " + title + " - Loss.png...")
    plt.savefig(title + " - Loss.png")
    plt.clf()


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    confs = [("LogLinear with w=0", 0), ("LogLinear with w=0.0001", 0.0001), ("LogLinear with w=0.001", 0.001)]
    data_manager = DataManager(ONEHOT_AVERAGE, batch_size=64)
    best_acc = 0
    best_results = None
    best_title = ""
    for (title, decay_rate) in confs:
        model = LogLinear(data_manager.get_input_shape()[0])
        results = train_model(model, data_manager, 20, 0.01, decay_rate)
        if results[3][-1] > best_acc:
            best_acc = results[3][-1]
            best_results = results
            best_title = title

        plots(title, *results[:4])

    print("Best model is " + best_title)
    print("Test Loss: %d Test Acc: %d" % (best_results[4], best_results[5]))
    print("Negated Polarity Acc: %d Rare Words Acc: %d" % (best_results[6], best_results[7]))


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    confs = [("LogLinearW2V with w=0", 0), ("LogLinearW2V with w=0.0001", 0.0001), ("LogLinearW2V with w=0.001", 0.001)]
    data_manager = DataManager(W2V_AVERAGE, batch_size=64, embedding_dim=300)
    best_acc = 0
    best_results = None
    best_title = ""
    for (title, decay_rate) in confs:
        model = LogLinear(data_manager.get_input_shape()[0])
        results = train_model(model, data_manager, 1, 0.01, decay_rate)
        if results[3][-1] > best_acc:
            best_acc = results[3][-1]
            best_results = results
            best_title = title

        plots(title, *results[:4])

    print("Best model is " + best_title)
    print("Test Loss: %d Test Acc: %d" % (best_results[4], best_results[5]))
    print("Negated Polarity Acc: %d Rare Words Acc: %d" % (best_results[6], best_results[7]))


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(W2V_SEQUENCE, batch_size=64, embedding_dim=300)
    model = LSTM(data_manager.get_input_shape()[1], 100, 1, 0.5)
    results = train_model(model, data_manager, n_epochs=4, lr=0.001, weight_decay=0.0001)

    plots("LSTMW2V with w=0.0001", *results[:4])

    print("Test Loss: %d Test Acc: %d" % (results[4], results[5]))


if __name__ == '__main__':
    # plots("LogLinear with w=0", w0_train_losses, w0_train_accs, w0_val_losses, w0_val_accs)
    # plots("LogLinear with w=0.0001", w1_train_losses, w1_train_accs, w1_val_losses, w1_val_accs)
    # plots("LogLinear with w=0.001", w2_train_losses, w2_train_accs, w2_val_losses, w2_val_accs)
    # train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    # train_lstm_with_w2v()
