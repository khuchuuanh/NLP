from lib import *
from process_data import *
from config import *

# RNN model
model_rnn = Sequential()
model_rnn.add(InputLayer(input_shape = (max_len_word,)))
model_rnn.add(Embedding(vocab_size_word, embedding_dim))
model_rnn.add(SimpleRNN(hidden_size, return_sequences= True))
model_rnn.add(TimeDistributed(Dense(vocab_size_tag)))
model_rnn.add(Activation('softmax'))

model_rnn.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

model_rnn.summary()


history_rnn = model_rnn.fit(train_padded_seqs, train_padded_tags,
                            batch_size = batch_size, epochs = epochs,
                            validation_data = (valid_padded_seqs, valid_padded_tags))


# LSTM model

model_lstm = Sequential()
model_lstm.add(InputLayer(input_shape = (max_len_word,)))
model_lstm.add(Embedding(vocab_size_word, embedding_dim))
model_lstm.add(LSTM(hidden_size, return_sequences= True))
model_lstm.add(TimeDistributed(Dense(vocab_size_tag)))
model_lstm.add(Activation('softmax'))

model_lstm.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

model_lstm.summary()

history_lstm = model_lstm.fit(train_padded_seqs, train_padded_tags,
                            batch_size = batch_size, epochs = epochs,
                            validation_data = (valid_padded_seqs, valid_padded_tags))



# GRU Model

model_gru = Sequential()
model_gru.add(InputLayer(input_shape = (max_len_word,)))
model_gru.add(Embedding(vocab_size_word, embedding_dim))
model_gru.add(GRU(hidden_size, return_sequences= True))
model_gru.add(TimeDistributed(Dense(vocab_size_tag)))
model_gru.add(Activation('softmax'))

model_gru.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

model_gru.summary()

history_gru = model_gru.fit(train_padded_seqs, train_padded_tags,
                            batch_size = batch_size, epochs = epochs,
                            validation_data = (valid_padded_seqs, valid_padded_tags))



# Bidirectional GRU

model_bigru = Sequential()
model_bigru.add(InputLayer(input_shape = (max_len_word,)))
model_bigru.add(Embedding(vocab_size_word, embedding_dim))
model_bigru.add(Bidirectional(GRU(hidden_size, return_sequences= True)))
model_bigru.add(TimeDistributed(Dense(vocab_size_tag)))
model_bigru.add(Activation('softmax'))

model_bigru.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

model_bigru.summary()

history_bigru = model_bigru.fit(train_padded_seqs, train_padded_tags,
                            batch_size = batch_size, epochs = epochs,
                            validation_data = (valid_padded_seqs, valid_padded_tags))


# Stack-GRU

model_tackgru = Sequential()
model_tackgru.add(InputLayer(input_shape = (max_len_word,)))
model_tackgru.add(Embedding(vocab_size_word, embedding_dim))
model_tackgru.add(GRU(hidden_size, return_sequences= True))
model_tackgru.add(GRU(hidden_size, return_sequences= True))
model_tackgru.add(TimeDistributed(Dense(vocab_size_tag)))
model_tackgru.add(Activation('softmax'))

model_tackgru.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

model_tackgru.summary()

history_model_tackgru= model_tackgru.fit(train_padded_seqs, train_padded_tags,
                            batch_size = batch_size, epochs = epochs,
                            validation_data = (valid_padded_seqs, valid_padded_tags))

