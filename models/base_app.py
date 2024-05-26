import json
import os
import argparse
from models.dot_dict import *
from models.torch_trainer import TorchTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from string import Template


DEFAULT_CONFIG_PATH = "./conf/base.json"


'''
    BaseApp 顶层APP实现类，主要实现一些解析参数的功能
    命令行参数优先级 > 用户指定的json_path > 默认的json_path
'''
class BaseApp(object):

    def __init__(self,json_path):
        self.init_args(json_path=json_path)
        self.trainer = TorchTrainer()

    def init_args(self,json_path):
        shell_args =  self.args_parse_from_shell()
        self.parse_from_json_file(shell_args=shell_args,json_path=json_path)
        self.check_cuda()
        self.replace_template()
        self.args = DotDict(self.args)

    def parse_from_json_file(self,shell_args,json_path):
        if len(shell_args.file) != 0:
            json_path = shell_args.file
        assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
        with open(DEFAULT_CONFIG_PATH,"r") as f:
           base_config = json.load(f)
        with open(json_path, 'r') as f:
            class_config = json.load(f)
        for key in class_config.keys():
            base_config[key] = class_config[key]
        self.args = base_config

    def args_parse_from_shell(self):
        parser = argparse.ArgumentParser(description='命令行参数改写json配置')
        # 添加命令行参数
        parser.add_argument('-f', '--file', help='指定文件路径',default="")  # 可选的参数
        return parser.parse_args()

    def check_cuda(self):
        # Check CUDA
        if not torch.cuda.is_available:
            self.args["cuda"] = False
        self.args["device"] = torch.device("cuda " if self.args["cuda"] else "cpu")
    
    def replace_template(self):
        for key in self.args:
            if type(self.args[key]) == str:
                template = Template(self.args[key])
                self.args[key] = template.substitute(self.args)

    def run(self,model,train_iter, eval_iter,args):
        for run_type in args.run_types:
            if run_type == "train":
                self.trainer.train(model=model,train_iter=train_iter,eval_iter=eval_iter,args=args)
            elif run_type == "test":
                self.trainer.test()