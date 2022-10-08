# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:03:28 2022

@author: zlf
"""
import numpy as np
import qutip as Q
from qutip import *
import pandas as pd
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import function_2_qubit
from function_2_qubit import *
import sys
sys.path.append("C:\\Users\\WIN10\\Desktop\\semi-supervised entanglement work")
import Generic_functions
from Generic_functions import *
number3=3000#
Number1=[500]
Number2=[10000]
#number1:The number of label states; number2:The number of unlabel states
K=2
M=2
dataset_2qubit(Number1,Number2,K) #Training set
x_val,y_val=data_generate(number3).Get_states() #Validation set
aug_val=Augmentation_Strategies(x_val,1)
x_val=aug_val.Augment()
y_val=aug_val.QB(y_val)
y_v=np.argmax(y_val,axis=1)
index_vs=np.argwhere(y_v == 0)
index_ve=np.argwhere(y_v == 1)
x_val=Feature_Trans(x_val).pauli_express()
augment=Augmentation_Strategies(K=K)
#%%
for (number1,number2) in zip(Number1,Number2):
    label_datape=np.load('./Data/label_datape_K'+repr(K)+'_'+repr(number1)+'.npy')
    label=np.load('./Data/labelpe_K'+repr(K)+'_'+repr(number1)+'.npy')
    unlabel_datape=np.load('./Data/unlabel_datape_K'+repr(K)+'_'+repr(number2)+'.npy')
    epoch=100#supervised
    Epoch=200#semi-supervised
    threshould=0.98
    rratio=[]
    iteration=30
    start=-1
    stop=1
    x=np.arange(start,stop,abs(start-stop)/iteration)
    y=Gauss_fun(x)
    select_number=[]
    acc=pd.DataFrame(np.zeros((iteration+1,4)),columns=["vsacc","veacc","val_acc","eval"])
    pmodel= tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=[4,4]),  
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(512, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(256, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(128, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(16, activation='relu' ),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(2, activation='softmax' )
        ])
    pmodel.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
    pmodel.fit(label_datape,label, epochs=epoch ,validation_split=0,shuffle=True ,batch_size = int(number1/10))
    Up=pmodel.predict(unlabel_datape,batch_size = number1)
    bas = [Up[k:k + K+1] for k in range(0, len(Up),K+1)]
    ub_average=1/(K+1)*(np.sum(bas, axis=1))
    ub=np.argmax(ub_average,axis=1)
    #ub_average=Sharpen(ub_average,T=0.5)
    index=np.argwhere(ub_average>threshould)[:,0]
    print('选择样本数：',(K+1)*len(index))
    index=augment.Index(index)
    ub=to_categorical(ub, num_classes=None)
    ub=augment.QB(ub)
    ub=ub[index]
    pv1=pmodel.predict(x_val,batch_size =number1)
    pv1=np.argmax(pv1,axis=1)
    pvsacc=2*len(np.argwhere(pv1[index_vs]== 0))/len(pv1)
    pveacc=2*len(np.argwhere(pv1[index_ve]== 1))/len(pv1)
    acc.iloc[0,0]=pvsacc
    acc.iloc[0,1]=pveacc
    acc.iloc[0,2]=(pvsacc+pveacc)/2
    acc.iloc[0,3]=0.45*pvsacc+0.55*pveacc
    pmodel.save('./model_2qubit/pmodel'+'_'+repr(number1)+'_'+repr(number2)+'.h5')
    for jj in range(iteration): 
        iterationloss=[]
        alpha=y[jj]          
        print('iteration=',jj)
        X=np.concatenate((label_datape,unlabel_datape[index]), axis=0)
        Y=np.concatenate((label,ub), axis=0)
        label_size=len(label)
        LOSS=[]
        mmodel= tf.keras.models.Sequential(
            [tf.keras.layers.Flatten(input_shape=[4,4]),  
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(512, activation='relu' ),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(256, activation='relu' ),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(128, activation='relu' ),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(16, activation='relu' ),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(2, activation='softmax' )
            ])
        def crossentropy(y_true, y_pred,L_S=label_size,lamda=alpha):
               cc =tf.keras.losses.CategoricalCrossentropy()
               loss1 =cc(y_true[0:L_S],y_pred[0:L_S])
               loss2 =cc(y_true[L_S:],y_pred[L_S:] )
               loss=loss1+lamda*loss2
               return loss
        mmodel.compile(loss=crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])
      
        mmodel.fit(X,Y, epochs=Epoch ,validation_split=0 ,batch_size = len(Y))     
        pv1=mmodel.predict(x_val,batch_size =len(x_val))
        pv1=np.argmax(pv1,axis=1)
        pvsacc=2*len(np.argwhere(pv1[index_vs]== 0))/len(pv1)
        pveacc=2*len(np.argwhere(pv1[index_ve]== 1))/len(pv1)
        acc.iloc[jj+1,0]=pvsacc
        acc.iloc[jj+1,1]=pveacc
        acc.iloc[jj+1,2]=(pvsacc+pveacc)/2
        acc.iloc[jj+1,3]=0.45*pvsacc+0.55*pveacc 
        Up=mmodel.predict(unlabel_datape,batch_size = len(unlabel_datape))
        bas = [Up[k:k + K+1] for k in range(0, len(Up),K+1)]
        ub_average=1/(K+1)*(np.sum(bas, axis=1))
        ub=np.argmax(ub_average,axis=1)
        #ub_average=Sharpen(ub_average,T=0.5)
        index=np.argwhere(ub_average>threshould)[:,0]
        ratio=sum(ub[index])/len(ub[index]) 
        rratio.append(ratio)
        index=augment.Index(index)
        ub=to_categorical(ub, num_classes=None)
        ub=augment.QB(ub)
        ub=ub[index]
        print('选择样本数：',len(index))
        select_number.append(len(index))
        mmodel.save('./model_2qubit/mmodel_inter'+repr(jj)+'KK'+repr(K)+'+_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')
        if jj>=1 and (acc["eval"][jj]>acc["eval"][jj-1]):
            mmodel.save('./max_model/best_KK'+repr(K)+'+_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.h5')
        acc.to_csv('./Result/valacc_KK'+repr(K)+'_'+repr(number1)+'_'+repr(number2)+'_'+repr(threshould)+'.csv')