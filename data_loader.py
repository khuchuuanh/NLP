from lib import *

class DataLoader:
    CONFIG = {
        'bpe_model_suffix' : '.model',
        'bpe_vocab_suffix' : '.vocab',
        'bpe_result_suffix': '.sequences'
    }

    Dictionary = {
        'source' :{
            'token2idx' : None,
            'idx2token' : None,
        },
        'target' :{
            'token2idx' : None,
            'idx2token' : None,
        }
    }
    Load_SP = {
        'source_sp': None,
        'target_sp': None
    }
    def __init__(self, data_dir, save_data_dir, source_lang, target_lang, buffer_size = 1000, batch_size = 64, bpe_vocab_size = 35000,seq_max_len_source = 50, seq_max_len_target = 50):
        self.data_dir = data_dir
        self.save_data_dir = save_data_dir
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.bpe_vocab_size=  bpe_vocab_size
        self.seq_max_len = {
            'source' : seq_max_len_source,
            'target': seq_max_len_target
        }
        self.paths = {
            'source' : {
                'train' : os.path.join(self.data_dir, 'train.' + self.source_lang),
                'valid': os.path.join(self.data_dir, 'valid.' + self.source_lang),
                'bpe_prefix' : os.path.join(self.save_data_dir, self.source_lang + '.sedmented'),
            },
            'target' :{
                'train': os.path.join(self.data_dir, 'train.' + self.target_lang),
                'valid': os.path.join(self.data_dir, 'valid.' + self.target_lang),
                'test': os.path.join(self.data_dir, 'test.' + self.target_lang),
                'bpe_prefix' : os.path.join(self.save_data_dir, self.target_lang + '.sedmented'),
            }
        } 


    def fit_dataloader(self):
        if not(os.path.exists(self.save_data_dir)):
            print('Create Save Vocab  and BPE Model Folders')
            os.mkdir(self.save_data_dir)

        print('#1-Prepare Dataset')
        source_train_data = self.load_data(self.paths['source']['train'])
        source_valid_data = self.load_data(self.paths['source']['valid'])

        source_train_data = self.load_data(self.paths['target']['train'])
        source_valid_data = self.load_data(self.paths['target']['valid'])

        print('#2 Train BPE Tokenization')
        self.train_bpe_tokenization(self.paths['source']['train'], self.paths['source']['bpe_prefix'])
        self.train_bpe_tokenization(self.paths['target']['train'], self.paths['target']['bpe_prefix'])

        print('#3 - Load BPE Encoder')
        self.load_bpe_encoder()

        print('#4 -Encode Data')
        source_train_sequences = self.texts_to_sequences(
            self.segment_sentence_piece(
            source_train_data,
            self.paths['source']['bpe_prefix'] + self.CONFIG['bpe_model_suffix'],
            self.paths['source']['bpe_prefix'] +'.train' + self.CONFIG['bpe_result_suffix']
            ), 
            mode = 'source'
        )

        source_valid_sequences = self.text_to_sequences(
            self.segment_setence_piece(
            source_valid_data,
            self.paths['source']['bpe_prefix'] + self.CONFIG['bpe_model_suffix'],
            self.paths['source']['bpe_prefix'] +'.valid' + self.CONFIG['bpe_result_suffix']
            ),
            mode = 'source'
        )

        target_train_sequences = self.texts_to_sequences(
            self.segment_sentence_piece(
            source_train_data,
            self.paths['source']['bpe_prefix'] + self.CONFIG['bpe_model_suffix'],
            self.paths['source']['bpe_prefix'] +'.train' + self.CONFIG['bpe_result_suffix']
            ), 
            mode = 'target'
        )

        target_valid_sequences = self.text_to_sequences(
            self.segment_setence_piece(
            source_valid_data,
            self.paths['source']['bpe_prefix'] + self.CONFIG['bpe_model_suffix'],
            self.paths['source']['bpe_prefix'] +'.valid' + self.CONFIG['bpe_result_suffix']
            ),
            mode = 'target'

        )
        print('=> Source: train: {}, valid: {}'.format(len(source_train_sequences), len(source_valid_sequences)))
        print('=> Target: train: {}, valid: {}'.format(len(target_train_sequences), len(target_valid_sequences)))

        print('#5- Convert DataLoader')
        
        train_dataset  = self.convert_dataset(source_train_sequences, target_train_sequences)
        valid_dataset = self.convert_dataset(source_valid_sequences, target_valid_sequences)

        print('Finish!')

        return train_dataset, valid_dataset
    
    def load_data(self, path):
        print('=> Load data from {}'.format(path))
        with open(path, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        return lines
    
    def load_test_dataset(self, test_source_data_path=None, test_target_data_path=None):
        print('#1-Prepare Test Dataset')
        if (test_source_data_path == None) and (test_target_data_path == None):
            source_test_data = self.load_data(self.paths['source']['test'])
            target_test_data = self.load_data(self.paths['target']['test'])
        else:
            source_test_data = self.load_data(test_source_data_path)
            target_test_data = self.load_data(test_target_data_path)