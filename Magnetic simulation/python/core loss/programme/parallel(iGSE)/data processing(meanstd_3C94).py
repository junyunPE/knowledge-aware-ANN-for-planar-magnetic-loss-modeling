import pandas as pd
import math 
import numpy as np
import tensorflow as tf
import optuna
tf.random.set_seed(2)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from optuna.samplers import TPESampler


path='D:\\Magnetic simulation\\python\\core loss\\programme\\parallel(iGSE)\\data(3C94).csv'
temp=pd.read_csv(path,header=0)
data=np.array(temp)
x=data[:,1:5]
y=data[:,7]

fmean=np.mean(x[:,0])
fstd=np.std(x[:,0])
Bmean=np.mean(x[:,1])
Bstd=np.std(x[:,1])
Dmean=np.mean(x[:,2]) #np.max(x[:,2])
Dstd=np.std(x[:,2])
iGSEmean=np.mean(x[:,3]) #np.max(x[:,2])
iGSEstd=np.std(x[:,3])
Mmean=np.mean(y)
Mstd=np.std(y)

x[:,0]=(x[:,0]-fmean)/fstd
x[:,1]=(x[:,1]-Bmean)/Bstd
x[:,2]=(x[:,2]-Dmean)/Dstd
x[:,3]=(x[:,3]-iGSEmean)/iGSEstd
y=(y-Mmean)/Mstd

'''
temp=np.linspace(0, 938, 150,dtype=int)
x_train=x[temp,:]
y_train=y[temp]
'''
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=200,shuffle = True,random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8,random_state=2)
    
def create_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-2, log=True)
    input1=tf.keras.layers.Input(shape=[4,])#input1
    for i in range(n_layers):#add hidden layers
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 3, 20, log=True)
        if(i==0):
            hidden=Dense(num_hidden,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input1)
        else:
            hidden=Dense(num_hidden,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(hidden)
    output1=Dense(1,activation="linear",name="output1")(hidden)
    input2=tf.keras.layers.Input(shape=[4,])
    output2=Dense(1,activation="linear",name="output2")(input2)
    output=tf.keras.layers.add([output1, output2])
    model=Model(inputs=[input1,input2],outputs=[output])

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
    tf.random.set_seed(2)
    model,batch_size,epochs = create_model(trial)
    model.fit([x_train,x_train],y_train,batch_size=batch_size,epochs=epochs)
    pred_test_y=model.predict([x_test,x_test])
    pred_train_y=model.predict([x_train,x_train])
    temp1=pred_test_y.reshape(-1)*Mstd+Mmean
    temp2=y_test*Mstd+Mmean
    temp3=pred_train_y.reshape(-1)*Mstd+Mmean
    temp4=y_train*Mstd+Mmean


    temp6=x_test[:,3]*iGSEstd+iGSEmean
    temp7=x_train[:,3]*iGSEstd+iGSEmean

    temp1=temp6/(1-temp1)
    temp2=temp6/(1-temp2)
    temp3=temp7/(1-temp3)
    temp4=temp7/(1-temp4)

    err_test=abs(100*(abs(temp1-temp2))/(temp2))
    err_train=abs(100*(abs(temp3-temp4))/(temp4))
    err=max(np.max(err_test),np.max(err_train))
    return err


sampler = TPESampler(seed=2)
study = optuna.create_study(direction="minimize",sampler=sampler)
study.optimize(objective, n_trials=200, timeout=1200)

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

tf.random.set_seed(2)
model,batch_size,epochs=create_model(trial)
model.fit([x_train,x_train],y_train,batch_size=batch_size,epochs=epochs)
pred_test_y=model.predict([x_test,x_test])
pred_val_y=model.predict([x_val,x_val])
pred_train_y=model.predict([x_train,x_train])
temp1=pred_test_y.reshape(-1)*Mstd+Mmean
temp2=y_test*Mstd+Mmean
temp3=pred_val_y.reshape(-1)*Mstd+Mmean
temp4=y_val*Mstd+Mmean
temp5=pred_train_y.reshape(-1)*Mstd+Mmean
temp6=y_train*Mstd+Mmean

temp7=x_test[:,3]*iGSEstd+iGSEmean
temp8=x_val[:,3]*iGSEstd+iGSEmean
temp9=x_train[:,3]*iGSEstd+iGSEmean


temp1=temp7/(1-temp1)
temp2=temp7/(1-temp2)
temp3=temp8/(1-temp3)
temp4=temp8/(1-temp4)
temp5=temp9/(1-temp5)
temp6=temp9/(1-temp6)

err_test=abs(100*(abs(temp1-temp2))/(temp2))
err_val=abs(100*(abs(temp3-temp4))/(temp4))
err_train=abs(100*(abs(temp5-temp6))/(temp6))
print(np.max(err_test))
print(np.mean(err_test))
print(np.max(err_val))
print(np.mean(err_val))
print(np.max(err_train))
print(np.mean(err_train))


data=pd.DataFrame({'fs':x_test[:,0]*fstd+fmean,'Bm':x_test[:,1]*Bstd+Bmean,'D':x_test[:,2]*Dstd+Dmean,\
    'pred_test_y':temp1,'y_test':temp2,'iGSE':temp7,'prediction error':err_test,\
    'iGSE error':100*(abs(temp7-temp2))/(temp2)})
data.to_csv('D:\\Magnetic simulation\\python\\core loss\\programme\\parallel(iGSE)\\data_test(3C94200).csv')
data1=pd.DataFrame({'fs':x_val[:,0]*fstd+fmean,'Bm':x_val[:,1]*Bstd+Bmean,'D':x_val[:,2]*Dstd+Dmean,\
    'pred_val_y':temp3,'y_val':temp4,'iGSE':temp8,'prediction error':err_val,\
        'iGSE error':100*(abs(temp8-temp4))/(temp4)})
data1.to_csv('D:\\Magnetic simulation\\python\\core loss\\programme\\parallel(iGSE)\\data_val(3C94200).csv')



