import sys
sys.path.append(r"C:\tf")
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Normalization

def Tensorflow():
    file_path = "student_admission_data.txt"
    data = np.loadtxt(file_path, skiprows=1)
    x=data[:,0:2]
    y=data[:,2]
    norm=Normalization(axis=-1)
    norm.adapt(x)
    Xn=norm(x)
    Xt=np.tile(Xn,(1000,1))
    y=y.reshape(-1,1)
    Yt=np.tile(y,(1000,1))
    tf.keras.Input((2,))
    model=Sequential(
        [Dense(units=25,activation="sigmoid",name="layer1"),
         Dense(units=15,activation="sigmoid",name="layer2"),
         Dense(units=1,activation="sigmoid",name="layer3"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    )
    model.fit(
        Xt,Yt,epochs=10,
    )
    w1,b1=model.get_layer("layer1").get_weights()
    w2,b2=model.get_layer("layer2").get_weights()
    w3,b3=model.get_layer("layer3").get_weights()
    return (norm,w1,b1,w2,b2,w3,b3)

def sigmoid(z):
    f=(1/(1+(np.exp(-z))))
    return f

def model(x,w1,b1,w2,b2,w3,b3):
    a1=dense(x,w1,b1)
    a2=dense(a1,w2,b2)
    a3=dense(a2,w3,b3)
    f=a3
    return f
def dense(a_in,w,b):
    units=w.shape[1]
    a_out=np.zeros(units)
    g=sigmoid
    for i in range(units):
        w1=w[:,i]
        z=np.dot(a_in,w1)+b[i]
        a_out[i]=g(z)
    return a_out
def my_predict(x,w1,b1,w2,b2,w3,b3):
    m=x.shape[0]
    p=np.zeros(m)
    for i in range(m):
        p[i]=model(x[i],w1,b1,w2,b2,w3,b3)
    return p
def Actual_predict(p):
    p1=np.zeros(len(p))
    for i in range(len(p)):
        p1[i]=1 if p[i]>=0.5 else 0
    return p1
def main():
    norm,w1,b1,w2,b2,w3,b3=Tensorflow()
    x=np.array([[66.59,80.82],[59.93,92.8],[31.78,92.1],[37.55,84.59]])
    x1=norm(x)
    p=my_predict(x1,w1,b1,w2,b2,w3,b3)
    p_A=Actual_predict(p)
    print(p_A)

main()
    
    
        
