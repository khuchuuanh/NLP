from config import *
from lib import *
from data_process import *


model = load_model("Bi_GRU.h5")

def generate_text(input_text, num_next_words, model):
  for i in range(num_next_words):
    sequences = tokenizer.texts_to_sequences([input_text])[0]
    test_padded_setences= pad_sequences([sequences], maxlen = max_len, padding ='pre')
    predict_word = model.predict(test_padded_setences)
    predict_max_word = np.argmax(predict_word, axis =1)
    for word, index in tokenizer.word_index.items():
      if predict_max_word == index:
        output = word
        break
    input_text = input_text + " " + word
  return input_text
