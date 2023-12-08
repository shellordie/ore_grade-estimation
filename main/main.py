import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from utils import BasicTrainer,xy_to_loader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

config={"data":r"./data/ore_grade_data.csv"}

def normalization(df):
    for col in df.select_dtypes(include=np.number):
        if col !="%Fe":
            np_df=df[col].to_numpy()
            df[col]=abs(np_df-np_df.mean())/np_df.std()
    return df 

def train_val_test_split(df):
    trainset,testset=train_test_split(df,test_size=0.1,random_state=0,shuffle=True)
    trainning,valset=train_test_split(trainset,test_size=0.3,random_state=0,shuffle=True)
    return trainning,valset,testset

def X_y_split(df):
    X=df.drop("%Fe",axis=1)
    y=df["%Fe"]
    return X,y

def preprocessing(data):
    df=pd.read_csv(data)
    df=normalization(df)
    trainset,valset,testset=train_val_test_split(df)
    X_train,y_train=X_y_split(trainset)
    X_val,y_val=X_y_split(valset)
    X_test,y_test=X_y_split(testset)
    X_train,y_train,trainloader=xy_to_loader(X_train,y_train)
    X_val,y_val,valloader=xy_to_loader(X_val,y_val)
    X_test,y_test,testloader=xy_to_loader(X_test,y_test)
    return(X_train,y_train,X_val,y_val,X_test,y_test),(trainloader,valloader,testloader) 

class iron_grade_net(nn.Module):
    def __init__(self,layer_size1,layer_size2):
        super().__init__()
        self.fc1=nn.Linear(2,layer_size1)
        self.fc2=nn.Linear(layer_size1,layer_size2)
        self.fc3=nn.Linear(layer_size2,1)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


if __name__=="__main__":
    tensors,loaders=preprocessing(config["data"])
    device=torch.device("cuda")
    model=iron_grade_net(512,2048)
    trainner=BasicTrainer(model=model,
                                    trainloader=loaders[0],
                                    valloader=loaders[1],
                                    criterion=nn.L1Loss(),
                                    optimizer=optim.Adam(model.parameters()),
                                    model_name="ore_grade_net")
    model=trainner.train(device=device,patience=5,
                     epoch=3000,min_delta=20,verbose=1)
    model.to('cpu')
    model=model.eval()
    with torch.no_grad():
        y_train_pred=model(tensors[0])
        y_val_pred=model(tensors[2])
        y_test_pred=model(tensors[4])
        train_rmse=mean_squared_error(y_train_pred,tensors[1],squared=False)
        val_rmse=mean_squared_error(y_val_pred,tensors[3],squared=False)
        test_rmse=mean_squared_error(y_test_pred,tensors[5],squared=False)
        print("train rmse",train_rmse)
        print("val rmse",val_rmse)
        print("test rmse",test_rmse)



