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

    