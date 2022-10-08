# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:00:21 2022

@author: zlf
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils.np_utils import to_categorical
import function_3_qubit
from function_3_qubit import *
import sys
sys.path.append("C:\\Users\\WIN10\\Desktop\\semi-supervised entanglement work")
import Generic_functions
from Generic_functions import *
t=0
number3=6000
K=8
K2=2
number1=20
number2=500
label_data,label=generate(number1=number1).Ghz()
la_data=Augmentation_Strategies(data=label_data,K=K,n=3).Augment()
lA=Augmentation_Strategies(K=K,n=3).QB(label)
ghz1,a1,ghz2,a2,ghz3,a3=generate(number1=number3).Ghz(v=1)

Ghz1,Ghz2,Ghz3=generate().test_Ghz(ghz1,ghz2,ghz3)
#%%
unlabel_data=generate(number2=number2).Un_State(label_data,label)
un=Augmentation_Strategies(data=unlabel_data, K=K2,n=3).Augment()
#%%
epoch=100#监督
Epoch=150#半监督
threshould=0.99
rratio=[]
iteration=30
start=-1
stop=1
x=np.arange(start,stop,abs(start-stop)/iteration)
y=Gauss_fun(x)
select_number=[]
bestepoch=[]
acc=pd.DataFrame(np.zeros((iteration+1,9)),columns=["v_3s","v_2s","v_e","val_acc","g_3s","g_2s","g_e","g_acc","eval"])
augment=Augmentation_Strategies(K=K2,n=3)
pmodel= tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=[8,8]),  
    tf.keras.layers.Dense(512, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(256, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(128, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(16, activation='relu' ),
    tf.keras.layers.Dense(3, activation='softmax' )
    ])
pmodel.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
pmodel.fit(la_data,lA, epochs=epoch ,validation_split=0,shuffle=True ,batch_size = 2000)
Up=pmodel.predict(un,batch_size = 2000)
bas = [Up[k:k + K2+1] for k in range(0, len(Up),K2+1)]
ub_average=1/(K2+1)*(np.sum(bas, axis=1))
ub=np.argmax(ub_average,axis=1)
#ub_average=Sharpen(ub_average,T=0.5)
index=np.argwhere(ub_average>threshould)[:,0]
print('选择样本数：',(K2+1)*len(index))
select_number.append((K2+1)*len(index))
index=augment.Index(index)
ub=to_categorical(ub, num_classes=3)
ub=augment.QB(ub)
ub=ub[index]
acc1=pmodel.evaluate(Ghz1,a1)[1]
acc2=pmodel.evaluate(Ghz2,a2)[1]
acc3=pmodel.evaluate(Ghz3,a3)[1]
acc.iloc[0,0]=acc1
acc.iloc[0,1]=acc2
acc.iloc[0,2]=acc3
acc.iloc[0,3]=(acc1+acc2+acc3)/3
b1=pmodel.evaluate(ghz1,a1)[1]
b2=pmodel.evaluate(ghz2,a2)[1]
b3=pmodel.evaluate(ghz3,a3)[1]
acc.iloc[0,4]=b1
acc.iloc[0,5]=b2
acc.iloc[0,6]=b3
acc.iloc[0,7]=(b1+b2+b3)/3
acc.iloc[0,8]=0.5*acc.iloc[0,7]+0.5*acc.iloc[0,3]
pmodel.save('./model_3qubit/a'+repr(t)+'_pmodel_'+'K'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')
for jj in range(iteration): 
    alpha=y[jj]          #(1/((np.pi*0.4)**(1/2)))*(math.exp( (-x[jj]**2)/0.4 ))
    print('iteration=',jj)
    X=np.concatenate((la_data,un[index]), axis=0)
    Y=np.concatenate((lA,ub), axis=0)
    label_size=len(la_data)
    mmodel= tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=[1,8,8]),  
        tf.keras.layers.Dense(512, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(256, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(128, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(16, activation='relu' ),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(3, activation='softmax' )
    ])
    def mycrossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):#batch_size标签数据
           cc =tf.keras.losses.CategoricalCrossentropy()
           loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
           loss2 =cc(y_true[L_S:],y_pred[L_S:] )
           loss=loss1+lamda*loss2
           return loss
    mmodel.compile(loss=mycrossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
    mmodel.fit(X,Y,epochs=Epoch ,validation_split=0,batch_size = len(Y))    
    acc1=mmodel.evaluate(Ghz1,a1)[1]
    acc2=mmodel.evaluate(Ghz2,a2)[1]
    acc3=mmodel.evaluate(Ghz3,a3)[1]
    acc.iloc[jj+1,0]=acc1
    acc.iloc[jj+1,1]=acc2
    acc.iloc[jj+1,2]=acc3
    acc.iloc[jj+1,3]=(acc1+acc2+acc3)/3
    b1=mmodel.evaluate(ghz1,a1)[1]
    b2=mmodel.evaluate(ghz2,a2)[1]
    b3=mmodel.evaluate(ghz3,a3)[1]
    acc.iloc[jj+1,4]=b1
    acc.iloc[jj+1,5]=b2
    acc.iloc[jj+1,6]=b3
    acc.iloc[jj+1,7]=(b1+b2+b3)/3
    acc.iloc[jj+1,8]=0.5*acc.iloc[jj+1,7]+0.5*acc.iloc[jj+1,3]
    Up=mmodel.predict(un,batch_size = len(un))
    bas = [Up[k:k + K2+1] for k in range(0, len(Up),K2+1)]
    ub_average=1/(K2+1)*(np.sum(bas, axis=1))
    ub=np.argmax(ub_average,axis=1)
    #ub_average=Sharpen(ub_average,T=0.5)
    index=np.argwhere(ub_average>threshould)[:,0]
    print('选择样本数：',(K2+1)*len(index))
    select_number.append((K2+1)*len(index))
    index=augment.Index(index)
    ub=to_categorical(ub, num_classes=3)
    ub=augment.QB(ub)
    ub=ub[index]
    mmodel.save('./model_3qubit/a'+repr(t)+'_mmodel_'+'KK'+repr(K)+'_'+repr(jj)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')
    if jj>=1 and (acc["eval"][jj]>acc["eval"][jj-1]):
        mmodel.save('./max_model/best_KK'+repr(K)+'+_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5') 
    acc.to_csv('./Result/a'+repr(t)+'_valacc_kK'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.csv')

