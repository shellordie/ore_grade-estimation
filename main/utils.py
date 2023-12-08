import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True



def xy_to_loader(X,y,batch_size=4,shuffle=False,return_tensor=True):
    X=torch.from_numpy(X.to_numpy()).float()
    y=np.reshape(y.to_numpy(),(y.shape[0],1))
    y=torch.from_numpy(y).float()
    Xy_set=TensorDataset(X,y)
    Xy_loader=DataLoader(Xy_set,batch_size,shuffle)
    if return_tensor==False:
        return Xy_loader 
    elif return_tensor==True:
        return X,y,Xy_loader


class BasicTrainer:
    def __init__(self,model,trainloader,valloader,
                 criterion,optimizer,model_name):
        self.model=model
        self.trainloader=trainloader
        self.valloader=valloader
        self.criterion=criterion
        self.optimizer=optimizer
        self.model_name=model_name
        self.model_path=self.create_dir()

    def create_dir(self):
        base_dir=os.getcwd()
        model_dir="{}/model".format(base_dir)
        self.model_path="{}/{}.pt".format(model_dir,self.model_name)
        if os.path.exists(model_dir)==False:
            os.mkdir(model_dir)
        return self.model_path

    def trainset_trainer(self,device):
        train_loss=0.0
        for i,data in enumerate(self.trainloader,0):
            inputs,labels=data[0].to(device),data[1].to(device)
            self.optimizer.zero_grad()
            # forward+backward+optimize
            outputs=self.model(inputs)
            loss=self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            train_loss+=loss.item()
        return train_loss

    def valset_trainer(self,device):
        val_loss=0.0
        for i,data in enumerate(self.valloader,0):
            inputs,labels=data[0].to(device),data[1].to(device)
            self.optimizer.zero_grad()
            # forward+backward+optimize
            outputs=self.model(inputs)
            loss=self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            val_loss+=loss.item()
        return val_loss

    def train (self,device,epoch=10,patience=3,min_delta=10,verbose=1):
        self.model=self.model.to(device)
        self.epoch=epoch
        self.verbose=verbose
        self.patience=patience
        self.min_delta=min_delta
        self.counter=0
        self.min_val_loss=float('inf')
        if self.verbose==2:print("training...")
        for epoch in range(self.epoch):
            train_loss=self.trainset_trainer(device)
            val_loss=self.valset_trainer(device)
            if self.verbose==1:
                print("[epoch {}]: train_loss={}||val_loss={}".format(epoch,train_loss,val_loss))
            if val_loss<self.min_val_loss:
                self.min_val_loss=val_loss
                torch.save(self.model.state_dict(),self.model_path)
                self.counter=0
            elif (val_loss>self.min_val_loss+self.min_delta):
                self.counter+=1
                if self.counter>=self.patience:
                    break
                else:
                    pass
        if self.verbose==2: print("[epoch {}]: train_loss={}||val_loss={}".format(epoch,train_loss,val_loss))
        return self.model

