from models.base_app import BaseApp
from utils.common import *
import torch.optim as optim
from tqdm import tqdm
from model import TextCNN

from dataset import ThucNewsDataSet

class APP(BaseApp):
    def __init__(self,json_path) -> None:
        super().__init__(json_path)
    
    def run(self):
        # 加载数据集
        data_set = ThucNewsDataSet(args=self.args) 
        self.model = TextCNN(class_num=len(data_set.LABEL.vocab),filter_sizes=self.args.filter_sizes,
                             filter_num=self.args.filter_num,vocabulary_size=len(data_set.TEXT.vocab),
                            embedding_dimension=data_set.TEXT.vocab.vectors.size()[-1],vectors=data_set.TEXT.vocab.vectors,dropout=self.args.dropout)
        super().run(self.model,data_set.train_iter, data_set.eval_iter,self.args)

if __name__ == "__main__":
    app = APP("./conf/text_cnn_classifier.json")
    app.run()