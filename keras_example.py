import numpy
import sys
# sys.path.append('/User/Reverie/keras/')

# Obtain the corpus of character sequence to train from.
# Here it is just the sequence 123456789 repeated 100000 times.
x = "123456789"*100000

# Construct a dictionary, and the reverse dictionary for the participating chars.
# '*" is a 'start-sequence' character.
dct = ['*'] + list(set(x))
max_features = len(dct)
rev_dct = [(j, i) for i, j in enumerate(dct)]
rev_dct = dict(rev_dct)

# Convert the characters to their dct indexes.
x = [rev_dct[ch] for ch in x]

# Divide the corpuse to substrings of length 200.
n_timestamps = 200
x = x[:len(x)- len(x) % n_timestamps]
x = numpy.array(x, dtype='int32').reshape((-1, n_timestamps))

# Generate input and ouput per substring, as an indicator matrix.
y = numpy.zeros((x.shape[0], x.shape[1], max_features), dtype='int32')
for i in numpy.arange(x.shape[0]):
    for j in numpy.arange(x.shape[1]):
        y[i, j, x[i, j]] = 1

# Shift-1 the input sequences to the right, and make them start with '*'.
x = numpy.roll(y, 1, axis=1)
x[:, 0, :] = 0
x[:, 0, 0] = 1

# Build the model.
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM

model = Sequential()
model.add(LSTM(max_features, 256, return_sequences=True))
model.add(LSTM(256, 256, return_sequences=True))
model.add(LSTM(256, 256, return_sequences=True))
model.add(TimeDistributedDense(256, max_features))
model.add(Activation('time_distributed_softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(x, y, batch_size=64, nb_epoch=50)

# Sample 128 sentences (200 characters each) from model.

def mnrnd(probs):
    rnd = numpy.random.random()
    for i in xrange(len(probs)):
        rnd -= probs[i]
        if rnd <= 0:
            return i
    return i

sentences = numpy.zeros((128, n_timestamps+1, max_features))
sentences[:, 0, 0] = 1

# Start sampling char-sequences. At each iteration i the probability over
# the i-th character of each sequences is computed.
for i in numpy.arange(n_timestamps):
    probs = model.predict_proba(sentences)[:,i,:]
    # Go over each sequence and sample the i-th character.
    for j in numpy.arange(len(sentences)):
        sentences[j, i+1, mnrnd(probs[j, :])] = 1
sentences = [sentence[1:].nonzero()[1] for sentence in sentences]

# Convert to readable text.
text = []
for sentence in sentences:
    text.append(''.join([dct[word] for word in sentence]))