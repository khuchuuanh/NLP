from lib import *


# load data
tagged_sentences = nltk.corpus.treebank.tagged_sents()
sentences, sentence_tags = [],[]
for tagged_sentence in tagged_sentences:
  sentence, tags = zip(*tagged_sentence)
  sentences.append(['startseq'] +[word.lower() for word in sentence] + ['endseq'])
  sentence_tags.append(['startseq'] + [tag for tag in tags] + ['endseq'])


# train test split
train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences,sentence_tags,
                                                                          test_size= 0.3)

valid_sentences, test_sentences, valid_tags, test_tags = train_test_split(test_sentences,test_tags,
                                                                          test_size= 0.5)


# preprocessing data
word_tokenizer = Tokenizer(oov_token = "<OOV>")
word_tokenizer.fit_on_texts(train_sentences)
train_seqs = word_tokenizer.texts_to_sequences(train_sentences)
valid_seqs = word_tokenizer.texts_to_sequences(valid_sentences)
max_len_word = np.max([len(seq) for seq in train_seqs])
vocab_size_word = len(word_tokenizer.index_word)+1


# padding or truncating
train_padded_seqs = pad_sequences(train_seqs, maxlen = max_len_word, padding = 'post')
valid_padded_seqs = pad_sequences(valid_seqs, maxlen = max_len_word, padding = 'post')



# processing for target
tag_tokenizer = Tokenizer(oov_token = '<OOV>')
tag_tokenizer.fit_on_texts(train_tags)
train_tags = tag_tokenizer.texts_to_sequences(train_tags)
valid_tags = tag_tokenizer.texts_to_sequences(valid_tags)

max_len_tag =np.max([len(seq) for seq in train_tags])


# padding and truncating for target
vocab_size_tag =len(tag_tokenizer.index_word) +1
train_padded_tags = pad_sequences(train_tags, maxlen = max_len_word, padding = 'post')
valid_padded_tags = pad_sequences(valid_tags, maxlen = max_len_word, padding = 'post')


# convert to one hot encode
train_padded_tags = to_categorical(train_padded_tags, num_classes =vocab_size_tag )
valid_padded_tags =  to_categorical(valid_padded_tags, num_classes =vocab_size_tag )