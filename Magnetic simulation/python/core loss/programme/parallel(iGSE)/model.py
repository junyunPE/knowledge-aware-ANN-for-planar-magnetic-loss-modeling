import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
n_layers = 1
weight_decay = 0.01
input1=tf.keras.layers.Input(shape=[2,])#input1
for i in range(n_layers):#add hidden layers
    num_hidden = 5
    if(i==0):
        hidden=Dense(num_hidden,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input1)
    else:
        hidden=Dense(num_hidden,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(hidden)
output1=Dense(1,activation="linear",name="output1")(hidden)
input2=tf.keras.layers.Input(shape=[1,])
output2=Dense(1,activation="linear",name="output2")(input2)
output=tf.keras.layers.add([output1, output2])
model=Model(inputs=[input1,input2],outputs=[output])
model.summary
from tensorflow.keras.utils import plot_model
plot_model(model,to_file='gd.png',show_shapes=True)