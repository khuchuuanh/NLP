from lib import *

class DataPreparing:
    def __init__(self, save_data_dir, source_lang, target_lang):
        self.save_data_dir = save_data_dir
        self.source_lang = source_lang
        self.target_lang = target_lang

    def download_dataset(self):
        if not(os.path.exists(self.save_data_dir)):
            print('Create Folder')
            os.mkdir(self.save_data_dir)

        if len(os.listdir(self.save_data_dir)) == 0:
            print('#1-Download Dataset')
            corpus = datasets.load_dataset('mt_eng_vietnamese', 'iwslt2015-en-vi')
            # corpus = datasets.load_datasets('wmt14','de-en')
            source_train, target_train = self.get_data(corpus['train'])
            source_valid, target_valid = self.get_data(corpus['validation'])
            source_test, target_test   = self.get_data(corpus['test'])

            print('Source lang : {} - train : {}, valid : {}, test : {}'.format(self.source_lang, len(source_train), len(source_valid), len(source_test)))
            print('Target lang: {} - train: {}, valid: {}, test: {}'.format(self.target_lang, len(target_train), len(target_valid), len(target_test)))

            print('#2 - Save dataset')
            self.save_data(source_train, os.path.join(self.save_data_dir, 'train.' + self.source_lang ))
            self.save_data(source_valid, os.path.join(self.save_data_dir, 'valid.' + self.source_lang))
            self.save_data(source_test, os.path.join(self.save_data_dir,'test.'+ self.source_lang))
            self.save_data(target_train, os.path.join(self.save_data_dir, 'train.' +self.target_lang))
            self.save_data(target_valid, os.path.join(self.save_data_dir, 'valid.' + self.target_lang))
            self.save_data(target_test, os.path.join(self.save_data_dir,'test.' + self.target_lang))
        
        else:
            print('Dataset exist!')

    def get_data(self, corpus):
        source_data = []
        target_data = []
        for data in corpus:
            source_data.append(self.preprocess_text(data['translation'][self.source_lang], self.source_lang))
            target_data.append(self.preprocess_text(data['translation'][self.target_lang], self.target_lang))
        return source_data, target_data

    def preprocess_text(self, text, lang = None):
        text = text.replace('&quot;', ' ')
        text = text.replace('&apos', "'")
        if lang == 'en':
            text = contractions.fix(text)
        text = text.lower()
        punc_number = list(string.punctuation + string.digits)
        for c in punc_number:
            text = text.replace(c, ' ')
        return  ' '.join(text.split())

    def save_data(self, data, save_path):
        print('=> Save data => Path : {}'.format(save_path))
        with open(save_path, 'w', encoding= 'utf-8') as f:
            f.write('\n'.join(data))

data_pre = DataPreparing('./save_data', 'vi','en')
data_pre.download_dataset()

