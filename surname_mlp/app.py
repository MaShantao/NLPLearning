from models.base_app import BaseApp
from utils.common import *
from surname_dataset import SurnameClassifer,SurnameVectorizer,SurnameDataSet
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

class APP(BaseApp):

    def __init__(self,json_path) -> None:
        super().__init__(json_path)
        set_seed_everywhere(self.args.seed,self.args.cuda)
        handle_dirs(self.args.save_dir)
    
    def create_mlp_and_dataset(self):
        if self.args.reload_from_files and os.path.exists(self.args.vectorizer_file):
            print("Reloading.....\n")
            self.dataset = SurnameDataSet.load_dataset_and_make_vectorizer(self.args.surname_csv,self.args.vectorizer_file)
        else:
            print("Creating Fresh!\n")
            self.dataset = SurnameDataSet.load_dataset_and_make_vectorizer(self.args.surname_csv)
            self.dataset.save_vectorizer(self.args.vectorizer_file)

        self.vectorizer = self.dataset.get_vectorizer()
        self.classifier = SurnameClassifer(input_dim=len(self.vectorizer.surname_vocab),hidden_dims=self.args.hidden_dims,out_dim=len(self.vectorizer.nationality_vocab))
        self.classifier = self.classifier.to(self.args.device)
    
    def generate_batches(self):
        dataloader = DataLoader(dataset=self.dataset,batch_size=self.args.batch_size,shuffle=self.args.shuffle,drop_last=self.args.drop_last)
        for data_dict in dataloader:
            out_data_dict ={}
            for name,tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(self.args.device)
            yield out_data_dict

    def make_train_state(self):
        return {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val': 1e8,
                'learning_rate':self.args.learning_rate,
                'epoch_index': 0,
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': -1,
                'test_acc': -1}

    def update_train_state(self,model,train_state):
        if train_state["epoch_index"] == 0:
            torch.save(model.state_dict(),self.args.model_state_file)
            train_state["stop_early"] = False
        elif train_state["epoch_index"] > 1:
            loss_tml,loss_t = train_state["val_loss"][-2:]
            if loss_t >= train_state["early_stopping_best_val"]:
                train_state["early_stopping_step"] +=1
            else:
                if loss_t < train_state["early_stopping_best_val"]:
                    torch.save(model.state_dict(),self.args.model_state_file)
                train_state["early_stopping_step"] = 0
            train_state["stop_early"] = train_state["early_stopping_step"] >=self.args.early_stopping_criteria

        return train_state

    def compute_accuracy(self,y_pred,y_target):
        _,y_pre_indices = y_pred.max(dim=1)
        n_corrtect = torch.eq(y_pre_indices,y_target).sum().item()
        return n_corrtect/len(y_pre_indices)*100
    
    def draw(self,train_state):
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
        fig.savefig("./surname_mlp/model_save/surname_mlp.jpg")

    def run(self):
        for run_type in self.args.run_types:
            if run_type == "train":
                train_state = self.train()
                self.draw(train_state)
            elif run_type == "test":
                self.test()
            elif run_type == "inference_topk_nationality":
                self.inference_topk_nationality()
            elif run_type == "inference_nationality":
                self.inference_nationality()

    def predict_nationality(self,surname,k=1):
        vectorized_surname = torch.tensor(self.vectorizer.vectorizer(surname))
        result = self.classifier(vectorized_surname.view(1, -1), apply_softmax=True)
        if k == 1:
            probability_values, indices = result.max(dim=1)
            index = indices.item()
            predicted_nationality = self.vectorizer.nationality_vocab.lookup_index(index)
            probability_value = probability_values.item()
            return {'nationality': predicted_nationality, 'probability': probability_value}
        elif k > 1:
            probability_values, indices = torch.topk(result, k=k)
            probability_values = probability_values.detach().numpy()[0]
            indices = indices.detach().numpy()[0]
            ret = []
            for prob_value, index in zip(probability_values, indices):
                nationality = self.vectorizer.nationality_vocab.lookup_index(index)
                ret.append({'nationality': nationality, 
                                'probability': prob_value})
            return ret
    
    def inference_nationality(self):
        new_surname = input("[inference_nationality] Enter a surname to classify: ")
        self.create_mlp_and_dataset()
        self.classifier.load_state_dict(torch.load(self.args.model_state_file))
        self.classifier = self.classifier.to(self.args.device)
        prediction = self.predict_nationality(new_surname)
        print("{} -> {} (p={:0.2f})".format(new_surname, prediction['nationality'],prediction['probability']))


    def inference_topk_nationality(self):
        new_surname = input("[inference_topk_nationality] Enter a surname to classify: ")
        self.create_mlp_and_dataset()
        self.classifier.load_state_dict(torch.load(self.args.model_state_file))
        self.classifier = self.classifier.to(self.args.device)
        k = int(input("How many of the top predictions to see? "))
        if k > len(self.vectorizer.nationality_vocab):
            print("Sorry! That's more than the # of nationalities we have.. defaulting you to max size :)")
            k = len(self.vectorizer.nationality_vocab)
        predictions = self.predict_nationality(new_surname,k=k)
        print("Top {} predictions:".format(k))
        print("===================")
        for prediction in predictions:
            print("{} -> {} (p={:0.2f})".format(new_surname,prediction['nationality'],prediction['probability']))

    def test(self):
        self.create_mlp_and_dataset()
        self.classifier.load_state_dict(torch.load(self.args.model_state_file))
        self.classifier = self.classifier.to(self.args.device)
        train_state = self.make_train_state()
        self.dataset.class_weights = self.dataset.class_weights.to(self.args.device)
        loss_func = nn.CrossEntropyLoss(self.dataset.class_weights)
        self.dataset.set_split('test')
        batch_generator = self.generate_batches()
        running_loss = 0.
        running_acc = 0.
        self.classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = self.classifier(x_in=batch_dict['x_surname'])
            loss = loss_func(y_pred, batch_dict['y_nationality'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            acc_t = self.compute_accuracy(y_pred, batch_dict['y_nationality'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            train_state['test_loss'] = running_loss
            train_state['test_acc'] = running_acc
            print("Test loss: {:.3f}".format(train_state['test_loss']))
            print("Test Accuracy: {:.2f}".format(train_state['test_acc']))

    def train(self):
        self.create_mlp_and_dataset()
        loss_func = nn.CrossEntropyLoss(self.dataset.class_weights)
        optimizer = optim.Adam(self.classifier.parameters(),lr = self.args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="min",factor=0.5,patience=1)

        train_state = self.make_train_state()
        epoch_bar = tqdm(desc = "traing_routine",total = self.args.num_epochs,position = 0)

        self.dataset.set_split("train")
        train_bar = tqdm(desc = "split=train",total = self.dataset.get_nums_batchs(self.args.batch_size),position = 1,leave = True)

        self.dataset.set_split("val")
        var_bar = tqdm(desc = "split=val",total = self.dataset.get_nums_batchs(self.args.batch_size),position = 2,leave = True)
        # Train
        try:
            for epoch_index in range(self.args.num_epochs):
                train_state["ecpoch_index"] = epoch_index
                self.dataset.set_split("train")
                batch_generator = self.generate_batches()
                running_loss = 0.0
                running_acc = 0.0
                self.classifier.train()
                for batch_index,batch_dict in enumerate(batch_generator):
                    optimizer.zero_grad()
                    y_pred = self.classifier(batch_dict["x_surname"])
                    loss = loss_func(y_pred,batch_dict["y_nationality"])
                    loss_t = loss.item()
                    running_loss +=(loss_t - running_loss)/(batch_index+1)
                    loss.backward()
                    optimizer.step()
                    acc_t = self.compute_accuracy(y_pred,batch_dict["y_nationality"])
                    running_acc +=(acc_t - running_acc)/(batch_index+1)
                    train_bar.set_postfix(loss = running_loss,acc = running_acc,epoch=epoch_index)
                    train_bar.update()
                train_state["train_loss"].append(running_loss)
                train_state["train_acc"].append(running_acc)

                self.dataset.set_split("val")
                batch_generator = self.generate_batches()

                running_loss = 0
                running_acc = 0
                self.classifier.eval()
                for batch_index,batch_dict in enumerate(batch_generator):
                    optimizer.zero_grad()
                    y_pred = self.classifier(batch_dict["x_surname"])

                    loss = loss_func(y_pred,batch_dict["y_nationality"])
                    loss_t = loss.to("cpu").item()
                    running_loss +=(loss_t - running_loss)//(batch_index+1)

                    acc_t = self.compute_accuracy(y_pred,batch_dict["y_nationality"])
                    running_acc +=(acc_t - running_acc)/(batch_index+1)
                    var_bar.set_postfix(loss = running_loss,acc = running_acc,epoch=epoch_index)
                    var_bar.update()

                train_state["val_loss"].append(running_loss)
                train_state["val_acc"].append(running_acc)

                train_state = self.update_train_state(model=self.classifier,train_state=train_state)
                scheduler.step(train_state["val_loss"][-1])

                if train_state["stop_early"]:
                    break
                train_bar.n=0
                var_bar.n=0
                epoch_bar.update() 
            return train_state
        except KeyboardInterrupt:
            print("KeyboardInterrupt \n")

if __name__ == "__main__":
    app = APP("./conf/surname_mlp.json")
    app.run()