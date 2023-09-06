import pandas as pd
import math 
import random
import numpy as np
import tensorflow as tf
import optuna
np.random.seed(42)
tf.random.set_seed(2)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from optuna.samplers import TPESampler


def run_optuna(x,y,model_name,n_trials, timeout):
    #Input and output data normalization
    x=(np.log(x) - np.min(np.log(x), axis=0)) / (np.max(np.log(x), axis=0) - np.min(np.log(x), axis=0))
    ymax=np.max(np.log(y))
    ymin=np.min(np.log(y))
    y=(np.log(y)-ymin)/(ymax-ymin)

    #Split data 
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.9,shuffle = True,random_state=2)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8,random_state=2)

    def create_model(trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-2, log=True)
        input1=tf.keras.layers.Input(shape=[3,])#input1
        for i in range(n_layers):#add hidden layers
            num_hidden = trial.suggest_int("n_units_l{}".format(i), 3, 20, log=True)
            if(i==0):
                hidden=Dense(num_hidden,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input1)
            else:
                hidden=Dense(num_hidden,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(hidden)
        output1=Dense(1,activation="linear",name="output1")(hidden)
        model=Model(inputs=[input1],outputs=[output1])

        kwargs = {}
        optimizer_options = ["RMSprop", "Adam","SGD"]
        optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
        if optimizer_selected == "RMSprop":
            kwargs["learning_rate"] = trial.suggest_float(
                "rmsprop_learning_rate", 1e-6, 1e-1, log=True
            )
        elif optimizer_selected == "Adam":
            kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-6, 1e-1, log=True)
        elif optimizer_selected == "SGD":
            kwargs["learning_rate"] = trial.suggest_float(
                "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
            )
            kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

        optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
        model.compile(optimizer,loss='mse')
        batch_size = trial.suggest_categorical('batch_size',[16,32,64,128,256,512])
        epochs=trial.suggest_int("epochs", 100, 500)
        return model,batch_size,epochs

    def objective(trial):
        random.seed(2)
        np.random.seed(2)
        tf.random.set_seed(2)
        model,batch_size,epochs = create_model(trial)
        model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs)
        pred_test_y=model.predict(x_test)
        pred_train_y=model.predict(x_train)
        temp1=np.exp(pred_test_y.reshape(-1)*(ymax-ymin)+ymin)
        temp2=np.exp(y_test*(ymax-ymin)+ymin)
        temp3=np.exp(pred_train_y.reshape(-1)*(ymax-ymin)+ymin)
        temp4=np.exp(y_train*(ymax-ymin)+ymin)
        err_test=abs(100*(abs(temp1-temp2))/(temp2))
        err_train=abs(100*(abs(temp3-temp4))/(temp4))
        err=max(np.max(err_test),np.max(err_train))
        return err

    sampler = TPESampler(seed=2)
    study = optuna.create_study(direction="minimize",sampler=sampler)
    study.optimize(objective, n_trials, timeout)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    random.seed(2)
    np.random.seed(2)
    tf.random.set_seed(2)
    model, batch_size, epochs = create_model(trial)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    pred_test_y=model.predict(x_test)
    pred_val_y=model.predict(x_val)
    pred_train_y=model.predict(x_train)
    temp1=np.exp(pred_test_y.reshape(-1)*(ymax-ymin)+ymin)
    temp2=np.exp(y_test*(ymax-ymin)+ymin)
    temp3=np.exp(pred_val_y.reshape(-1)*(ymax-ymin)+ymin)
    temp4=np.exp(y_val*(ymax-ymin)+ymin)
    temp5=np.exp(pred_train_y.reshape(-1)*(ymax-ymin)+ymin)
    temp6=np.exp(y_train*(ymax-ymin)+ymin)
    err_test=abs(100*(abs(temp1-temp2))/(temp2))
    err_val=abs(100*(abs(temp3-temp4))/(temp4))
    err_train=abs(100*(abs(temp5-temp6))/(temp6))
    print(np.max(err_train))
    print(np.mean(err_train))
    print(np.max(err_test))
    print(np.mean(err_test))
    print(np.max(err_val))
    print(np.mean(err_val))
    model.save(model_name)


path='C:\\Users\\junyun_deng\\Desktop\\data\\Rdc.csv'
temp=pd.read_csv(path,header=0)
data=np.array(temp)
x=data[:,0:3]
y=data[:,3]
run_optuna(x,y,'Rdc_model',n_trials=200, timeout=1200)


