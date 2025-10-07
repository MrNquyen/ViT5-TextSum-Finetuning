import torch
from collections import defaultdict
from torch import nn
from utils.utils import load_vocab
from icecream import ic

# Use to training tokenizer from scratch
class BaseVocab(nn.Module):
    def __init__(
            self, 
            vocab_file,
        ):
        """
            Vocab class to be used when you want to train word embeddings from
            scratch based on a custom vocab. This will initialize the random
            vectors for the vocabulary you pass. Get the vectors using
            `get_vectors` function. This will also create random embeddings for
            some predefined words like PAD - <pad>, SOS - <s>, EOS - </s>,
            UNK - <unk>.

            Parameters
            ----------
            vocab_file : str
                Path of the vocabulary file containing one word per line
            embedding_dim : int
                Size of the embedding

        """
        super().__init__()
        #-- 1. Init stoi and itos
        self.init_special_tokens()
        self.word_dict = {}
        self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX

        self.itos = {}
        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN

        self.vocabs = load_vocab(vocab_file)
        for i, token in enumerate(self.vocabs):
            token_idx = 4 + i
            self.word_dict[token] = token_idx
            self.itos[token_idx] = token

        self.stoi = defaultdict(lambda: self.UNK_INDEX, self.word_dict)

    #-- Build
    def init_special_tokens(self):
        self.PAD_TOKEN = "<pad>"
        self.SOS_TOKEN = "<s>"
        self.EOS_TOKEN = "</s>"
        self.UNK_TOKEN = "<unk>"

        self.PAD_INDEX = 0
        self.SOS_INDEX = 1
        self.EOS_INDEX = 2
        self.UNK_INDEX = 3


    #-- Get
    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def get_size(self):
        return len(self.itos)

    def get_pad_index(self):
        return self.PAD_INDEX

    def get_pad_token(self):
        return self.PAD_TOKEN

    def get_start_index(self):
        return self.SOS_INDEX

    def get_start_token(self):
        return self.SOS_TOKEN

    def get_end_index(self):
        return self.EOS_INDEX

    def get_end_token(self):
        return self.EOS_TOKEN

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def get_vectors(self):
        return getattr(self, "vectors", None)
    
    def get_word_idx(self, word):
        return self.stoi[word]
    
    def get_idx_word(self, idx):
        return self.itos[idx]
        
    

#------------------------------------------------
class CustomVocab(BaseVocab):
    def __init__(
            self, 
            tokenizer=None,
            vocab_file=None,
        ):
        """
        Use this vocab class when you have a custom vocabulary class but you
        want to use pretrained embedding vectos for it. This will only load
        the vectors which intersect with your vocabulary. 

        Parameters
        ----------
        vocab_file : str
            Vocabulary file containing list of words with one word per line
            which will be used to collect vectors
        """
        self.tokenizer = tokenizer
        super().__init__(
            vocab_file=vocab_file
        )
        # self.init_special_tokens()
    

    def init_special_tokens(self):
        self.PAD_TOKEN = self.tokenizer.pad_token
        self.SOS_TOKEN = self.tokenizer.cls_token
        self.EOS_TOKEN = self.tokenizer.eos_token
        self.UNK_TOKEN = self.tokenizer.unk_token

        self.PAD_INDEX = self.tokenizer.pad_token_id
        self.SOS_INDEX = self.tokenizer.cls_token_id
        self.EOS_INDEX = self.tokenizer.eos_token_id
        self.UNK_INDEX = self.tokenizer.unk_token_id