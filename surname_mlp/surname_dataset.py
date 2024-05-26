from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import json
import os
from argparse import Namespace

class Vocabulary(object):

    def __init__(self,token_to_idx =None,add_unk=True,unk_token="<UNK>"):
        if token_to_idx is None:
            token_to_idx={}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token,idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        return {"token_to_idx":self._token_to_idx,"add_unk":self._add_unk,"unk_token":self._unk_token}
    
    @classmethod
    def from_serializable(cls,contents):
        return cls(**contents)
    
    def add_token(self,token):
        try:
            index = self._token_to_idx[token]
        except KeyError:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
    
    def add_many(self,tokens):
        return [self.add_token(token) for token in tokens]
    
    def lookup_token(self,token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token,self.unk_index)
        else:
            return self._token_to_idx[token]
    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]
    def __str__(self) -> str:
        return "<Vocabulary(size=%d)>"%len(self)
    
    def __len__(self):
        return len(self._token_to_idx)


class SurnameVectorizer(object):

    def __init__(self, surname_vocab, nationality_vocab):
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab

    def vectorizer(self,surname):
        vacab = self.surname_vocab
        one_hot = np.zeros(len(vacab),dtype=np.float32)
        for token in surname:
            one_hot[vacab.lookup_token(token)] = 1
        return one_hot
    
    @classmethod
    def from_dataframe(cls,surname_df):
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        for index,row in surname_df.iterrows():
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)
        return cls(surname_vocab,nationality_vocab)
    
    @classmethod
    def from_serializable(cls,contents):
        surname_vocab = Vocabulary.from_serializable(contents["surname_vocab"])
        nationality_vocab = Vocabulary.from_serializable(contents["nationality_vocab"])
        return cls(surname_vocab = surname_vocab,nationality_vocab = nationality_vocab)

    def to_serializable(self):
        return {"surname_vocab":self.surname_vocab.to_serializable(),"nationality_vocab":self.nationality_vocab.to_serializable()}
    

class SurnameDataSet(Dataset):

    def __init__(self, surname_df,vectorizer):
        self.surname_df = surname_df
        self._vectorizer = vectorizer

        self.train_df = self.surname_df[self.surname_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.surname_df[self.surname_df.split=='val']
        self.val_size = len(self.val_df)

        self.test_df = self.surname_df[self.surname_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {"train":(self.train_df,self.train_size),
                             "val":(self.val_df,self.val_size),
                             "test":(self.test_df,self.test_size)}
        
        self.set_split("train")

        class_counts = surname_df.nationality.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.nationality_vocab.lookup_token(item[0])
        
        sorted_counts = sorted(class_counts.items(),key = sort_key)
        frequencies = [count for _,count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies,dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls,surname_csv,vectorizer_filepath):
        surname_df = pd.read_csv(surname_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(surname_df,vectorizer)
    
    def __len__(self):
        return self._target_size
    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):
        surname_df = pd.read_csv(surname_csv)
        train_surname_df = surname_df[surname_df.split == 'train']
        return cls(surname_df, SurnameVectorizer.from_dataframe(train_surname_df))

    @classmethod
    def load_vectorizer_only(cls,vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return SurnameVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self,vectorizer_filepath):
        with open(vectorizer_filepath,"w") as fp:
            json.dump(self._vectorizer.to_serializable(),fp)
    
    def get_vectorizer(self):
        return self._vectorizer
    
    def set_split(self,split = "train"):
        self._target_split = split
        self._target_df,self._target_size = self._lookup_dict[split]
    
    def __getitem__(self,index):
        row = self._target_df.iloc[index]
        surname_vector = self._vectorizer.vectorizer(row.surname)
        nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)
        return {"x_surname":surname_vector,"y_nationality":nationality_index}
    
    def get_nums_batchs(self,batch_size):
        return len(self)//batch_size
    

class SurnameClassifer(nn.Module):
    def __init__(self, input_dim,hidden_dims,out_dim) -> None:
        super(SurnameClassifer,self).__init__()
        self.layers = nn.ModuleList()
        layer_dims = [input_dim] + hidden_dims + [out_dim]
        # 使用 for 循环添加线性层
        for i in range(len(layer_dims)-1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                self.layers.append(nn.ReLU())
        print(self.layers)    
        

    
    def forward(self,x_in,apply_softmax = False):
        for layer in self.layers:
            x_in = layer(x_in)
        if apply_softmax:
            x_in = F.softmax(x_in,dim=1)
        return x_in