# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 18:27:21 2022

@author: zlf
"""
import numpy as np
import qutip as Q
from qutip import *
from keras.utils.np_utils import to_categorical
import sys
sys.path.append("C:\\Users\\WIN10\\Desktop\\semi-supervised entanglement work")
import Generic_functions
from Generic_functions import *
class data_generate():
    def __init__(self,number1=None,number2=None):
        self.number1=number1
        self.number2=number2
        self.R_entangle=np.zeros((4,4)).reshape(1,4,4)
        self.R_separable=np.zeros((4,4)).reshape(1,4,4)
        self.y=np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
    def detect(self,H):
        Alpha=np.dot(np.dot(np.dot(H,self.y),H.conjugate()),self.y)
        eigenvalue,featurevector=np.linalg.eig(Alpha)
        eigenvalue.sort()        
        e=[eigenvalue[0].real,eigenvalue[1].real,eigenvalue[2].real,eigenvalue[3].real]
        e=abs(np.array(e))
        C_Rho=np.max([0,(e[3])**(1/2)-(e[2])**(1/2)-(e[1])**(1/2)-(e[0])**(1/2)])
        return C_Rho
    def get_rho(self):#density matrix
        M=np.random.randint(-10,10,size=(4,4))
        N=np.random.randint(-10,10,size=(4,4))
        H=np.dot(M+1j*N,((M+1j*N).T).conjugate()) #H      
        Rho=(1/(np.trace(H)))*H 
        return Rho
    def Get_states(self): #label_data/test data
        L1=0.0
        number1=np.array(self.number1)
        while L1<number1:
            Rho=data_generate().get_rho() 
            C_Rho=data_generate().detect(Rho)
            if C_Rho>0.0000000000001:
                 self.R_entangle=np.concatenate((self.R_entangle,Rho.reshape(1,4,4)), axis=0)
            else:
                 self.R_separable=np.concatenate((self.R_separable,Rho.reshape(1,4,4)), axis=0)
                 L1+=2
        self.R_separable=self.R_separable[1:int(0.50*number1)+1,:,:]
        self.R_entangle=self.R_entangle[1:int(0.50*number1)+1,:,:]
        data=np.concatenate((self.R_separable,self.R_entangle), axis=0)
        Label=np.zeros((len(data),1))
        s=len(self.R_separable)
        Label[s:]=1 
        permutation1 = list(np.random.permutation(len(data)))
        self.shuffled_data1 = data[permutation1]
        shuffled_label1=Label[permutation1]
        self.shuffled_label1= to_categorical(shuffled_label1, num_classes=None)
        return  self.shuffled_data1,self.shuffled_label1
    def Unlabel_Gstates(self): 
        number2=self.number2[0]
        unlabel_data=np.zeros((4,4)).reshape(1,4,4)
        for i in range(number2):
           rho=data_generate().get_rho().reshape(1,4,4) 
           unlabel_data=np.concatenate((unlabel_data,rho), axis=0)
        unlabel_data=unlabel_data[1:,:,:]
        ratio=int((number2*0.7-number2*0.3)/len(self.shuffled_label1))
        itemindex_s = np.argwhere((self.shuffled_label1==[1,0]))#pick separable states
        itemindex_s=itemindex_s[np.arange(0,len(itemindex_s),2)][:,0]
        se1 = self.shuffled_data1[itemindex_s]      
        augmentation_strategies=Augmentation_Strategies(se1,ratio-1)
        seundata=augmentation_strategies.Augment(1)
        CCseundata=augmentation_strategies.Convex_Combination()
        unlabel_data=np.concatenate((seundata,CCseundata,unlabel_data), axis=0)
        permutation1 = list(np.random.permutation(len(unlabel_data)))
        unlabel_data=unlabel_data[permutation1]
        return unlabel_data[0:number2]
def dataset_2qubit(number1,number2, K): 
        get_data=data_generate(number1,number2)
        label_da,la=get_data.Get_states()
        np.save('./Data/label_data_'+repr(number1)+'.npy', label_da)
        np.save('./Data/label_'+repr(number1)+'.npy', la)
        undata=get_data.Unlabel_Gstates()
        np.save('./Data/unlabel_data_'+repr(number2)+'.npy', undata)
        label_data=Augmentation_Strategies(label_da,K).Augment()
        label_datape=Feature_Trans(label_data).pauli_express()
        label=Augmentation_Strategies().QB(la)
        permutation1 = list(np.random.permutation(len(label_datape)))
        label_datape=label_datape[permutation1]
        label=label[permutation1]
        np.save('./Data/label_datape_K'+repr(K)+'_'+repr(number1)+'.npy', label_datape)
        np.save('./Data/labelpe_K'+repr(K)+'_'+repr(number1)+'.npy', label)
        unlabel_data=Augmentation_Strategies(undata,K).Augment()
        unlabel_datape=Feature_Trans(unlabel_data).pauli_express()
        np.save('./Data/unlabel_datape_K'+repr(K)+'_'+repr(number2)+'.npy', unlabel_datape)