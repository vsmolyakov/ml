
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop

import numpy as np
import random
import sys

def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)

    return np.argmax(probs)


if __name__ == "__main__":

    path = './data/shakespeare.txt'
    text = open(path).read().lower()
    print "corpus length: ", len(text)

    chars = sorted(list(set(text)))
    print "total chars: ", len(chars)

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # split text into sentences
    step = 3
    maxlen = 40
    sentences, next_chars = [], []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i+maxlen])
        next_chars.append(text[i+maxlen])
    print "number of sentences: ", len(sentences)

    # vectorize
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # LSTM
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # LSTM training
    print "LSTM training..."
    epochs = 10 
    for i in range(epochs):
        print "epoch: ", i
        model.fit(X, y, batch_size=128, nb_epoch=1)
        
        # print out model generated text
        generated = ''
        start_index = random.randint(0, len(text)-maxlen-1)
        sentence = text[start_index: start_index + maxlen]
        generated += sentence

        for i in range(400):
            x = np.zeros((1,maxlen,len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
        
        print generated

       






 
