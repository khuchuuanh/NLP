from lib import *
from config import *

corpus = []
with open('truyen_kieu_data.txt', 'r', encoding="utf8") as f:
  corpus = f.readlines()


punct = string.punctuation + string.digits + "''" + "\n"

def clean_data(text):
    text = text.lower()
    text = text.replace('[<p style="text-align: center;">', "")
    text = text.replace('<br/>\n', "")
    text = text.replace('</p>, <p style="text-align: center;">', " ")
    text = text.replace('</p>]', "")
    text = text.translate(str.maketrans(" ", " ", punct))
    return text


x = set()
for line in corpus:
  list_word= line.split()
  for i in list_word:
    x.add(i)


corpus = [clean_data(i) for i in corpus]
for i in corpus:
    if len(i) == 0:
        corpus.remove(i)
        

train_sentences = []
train_labels = []

for line in corpus:
  list_word= line.split()
  for i in range(1, len(list_word)):
    sentence = list_word[:i]
    label = list_word[i]
    train_sentences.append(sentence)
    train_labels.append(label)



tokenizer = Tokenizer(num_words= vocab_size, oov_token = "<OOV>")
tokenizer.fit_on_texts(corpus)

num_word = len(tokenizer.word_index) +1


train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded_sequences = pad_sequences(train_sequences, maxlen= max_len, truncating = "pre", padding ="pre")

train_labels = tokenizer.texts_to_sequences(train_labels)
train_labels = ku.to_categorical(train_labels, num_classes = num_word)