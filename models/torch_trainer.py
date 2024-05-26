from utils.common import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm


class TorchTrainer(object):

    def train(self,model,args,train_iter,eval_iter,optim = None,loss_func = None):
        train_state = self.make_train_state(args)
        epoch_bar = tqdm(desc = "traing_routine",total = args.num_epochs,position = 0)
        train_bar = tqdm(desc = "split=train",total = len(train_iter),position = 1,leave = True)
        print("len(train_iter)",len(train_iter))
        var_bar = tqdm(desc = "split=val",total = len(eval_iter),position = 2,leave = True)
        print("len(eval_iter)",len(eval_iter))
        loss_func = self.loss_func(loss_func,args)
        optim = self.optimizer(optim,model,args)
        
        try:
            for epoch_index in range(args.num_epochs):

                # Train
                running_loss = 0.0
                running_acc = 0.0
                model.train()
                for batch_index,batch_data in enumerate(train_iter):
                    feature, target = batch_data.content, batch_data.label
                    if torch.cuda.is_available(): # 如果有GPU将特征更新放在GPU上
                        feature,target = feature.cuda(),target.cuda()
                    optim.zero_grad()
                    logits,output = model(feature)
                    loss = loss_func(logits,target)
                    loss_t = loss.item()
                    running_loss +=(loss_t - running_loss)/(epoch_index + 1)
                    loss.backward()
                    optim.step()
                    acc_t = self.compute_accuracy(target,output)
                    running_acc += (acc_t - running_acc) / (batch_index + 1)
                    train_bar.set_postfix(loss_func = running_loss,acc = running_acc,epoch=epoch_index)
                    train_bar.update()
                train_state["train_loss"].append(running_loss)
                train_state["train_acc"].append(running_acc)

                

                # Eval
                running_loss = 0
                running_acc = 0
                model.eval()
                for batch_index,batch_data in enumerate(eval_iter):
                    feature, target = batch_data.content, batch_data.label
                    if torch.cuda.is_available(): # 如果有GPU将特征更新放在GPU上
                        feature,target = feature.cuda(),target.cuda()
                    logits,output = model(feature)
                    loss = loss_func(logits,target)
                    loss_t = loss.item()
                    running_loss +=(loss_t - running_loss)/(epoch_index + 1)
                    acc_t = self.compute_accuracy(target,output)
                    running_loss +=(loss_t - running_loss)//(batch_index+1)
                    running_acc += (acc_t - running_acc) / (batch_index + 1)
                    var_bar.set_postfix(loss_func = running_loss,acc = running_acc,epoch=epoch_index)
                    var_bar.update()
                train_state["val_loss"].append(running_loss)
                train_state["val_acc"].append(running_acc)

                # Update Train State
                train_state = self.update_train_state(model=model,train_state=train_state,args = args)
                
                train_bar.n=0
                var_bar.n=0
                epoch_bar.update()

            return train_state
        except KeyboardInterrupt:
            print("KeyboardInterrupt \n")
        
        self.draw_train_fig(train_state,args)


    def test(self):
        return

    def loss_func(self,loss_func,args):
        if loss_func is not None:
            return loss_func
        elif args.loss_func == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()

    def optimizer(self,optim,model,args):
        if optim is not None:
            return optim
        elif args.optimizer == "Adam":
            return  torch.optim.Adam(model.parameters(), lr=args.learning_rate) # 梯度下降优化器，采用Adam

    def compute_accuracy(self,targets,outputs):
        total = targets.size(0)
        correct = (outputs == targets).sum().item()
        accuracy = correct / total
        return accuracy

    def make_train_state(self,args):
        return {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val': 1e8,
                'learning_rate':args.learning_rate,
                'epoch_index': 0,
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': -1,
                'test_acc': -1}
    
    def draw_train_fig(self,train_state,args):
        epochs = range(0, len(train_state['train_loss']))
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(121)
        ax.plot(epochs, train_state['train_loss'], label="train_loss")
        ax.plot(epochs, train_state['val_loss'], label="val_loss")
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(epochs, train_state['train_acc'], label="train_acc")
        ax.plot(epochs, train_state['val_acc'], label="val_acc")
        ax.legend()
        fig.savefig(args.train_fig)
    
    def update_train_state(self,args,model,train_state):
        if train_state["epoch_index"] == 0:
            torch.save(model.state_dict(),os.path.join(args.save_dir, args.model_name))
            train_state["stop_early"] = False
        elif train_state["epoch_index"] > 1:
            loss_tml,loss_t = train_state["val_loss"][-2:]
            if loss_t >= train_state["early_stopping_best_val"]:
                train_state["early_stopping_step"] +=1
            else:
                if loss_t < train_state["early_stopping_best_val"]:
                    torch.save(model.state_dict(),os.path.join(args.save_dir, args.model_name))
                train_state["early_stopping_step"] = 0
            train_state["stop_early"] = train_state["early_stopping_step"] >= args.early_stopping_criteria

        return train_state