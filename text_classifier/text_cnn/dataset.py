import torch
from torch.utils.data import Dataset
import jieba
import logging
import torchtext
from enum import Enum
from models.base_app import BaseApp
from tqdm import tqdm
jieba.setLogLevel(logging.INFO)

class TrainState(Enum):
    TRAIN = 1
    EVAL = 2
    TEST = 3

class ThucNewsDataSet(Dataset):
    
    def __init__(self, args):
        self.run_type = TrainState.TRAIN
        # 分词函数
        def cut(sentence):
            stopwords = open(args.stopwords_path).read().split('\n')
            return [token for token in jieba.lcut(sentence) if token not in stopwords]
     
        self.TEXT = torchtext.data.Field(sequential=True,lower=True,tokenize=cut)
        self.LABEL = torchtext.data.LabelField(sequential=False, dtype=torch.int64)
        self.train_dataset,self.dev_dataset,self.test_dataset = tqdm(
            torchtext.data.TabularDataset.splits(
            path=args.data_set.data_path,                 #文件存放路径
            format=args.data_set.data_format,                  #文件格式
            skip_header=False,             #是否跳过表头，我这里数据集中没有表头，所以不跳过
            train=args.data_set.train_name,  
            validation=args.data_set.val_name,
            test=args.data_set.test_name,    
            fields=[('label',self.LABEL),('content',self.TEXT)] # 定义数据对应的表头
            ),
            total=3
        )
        vectors = torchtext.vocab.Vectors(name=args.pretrained_name, 
            cache=args.pretrained_path)
        
        self.TEXT.build_vocab(self.train_dataset, self.dev_dataset,self.test_dataset,vectors=vectors)
        self.LABEL.build_vocab(self.train_dataset, self.dev_dataset,self.test_dataset)

        self.train_iter, self.eval_iter,self.test_iter = torchtext.data.BucketIterator.splits(
            (self.train_dataset, self.dev_dataset,self.test_dataset),   #需要生成迭代器的数据集
            batch_sizes=(args.batch_size, args.batch_size,args.batch_size),                  # 每个迭代器分别以多少样本为一个batch,验证集和测试集数据不需要训练，全部放在一个batch里面就行了
            sort_key=lambda x: len(x.content)            #按什么顺序来排列batch，这里是以句子的长度，就是上面说的把句子长度相近的放在同一个batch里面
        )

    def train(self):
        self.run_type = TrainState.TRAIN

    def eval(self):
        self.run_type = TrainState.EVAL

    def test(self):
        self.run_type = TrainState.TEST    
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
if __name__ == "__main__":
    app = BaseApp("./conf/text_cnn_classifier.json")
    dataset = ThucNewsDataSet(app.args)