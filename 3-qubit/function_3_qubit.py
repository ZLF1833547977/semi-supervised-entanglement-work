# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:10:22 2022

@author: zlf
"""
import numpy as np
import random 
I=np.eye(8)
import sys
sys.path.append("C:\\Users\\WIN10\\Desktop\\semi-supervised entanglement work")
import Generic_functions
from Generic_functions import *
class generate():
    def __init__(self,number1=None,number2=None,n=3):
        self.number1=number1
        self.number2=number2
        self.n=n
        self.Ghz1=np.zeros((2**self.n,2**self.n)).reshape(1,2**self.n,2**self.n)
        self.Ghz2=np.zeros((2**self.n,2**self.n)).reshape(1,2**self.n,2**self.n)
        self.Ghz3=np.zeros((2**self.n,2**self.n)).reshape(1,2**self.n,2**self.n)
    def Ghz(self,N=[1/5,1/5,3/7,3/7],v=0): 
        ghz_3=((1/np.sqrt(2))*(I[0,:]+I[2**self.n-1,:])).reshape(2**self.n,1)
        for i in range(int(self.number1/3)):
            g1=random.uniform(0,N[0])
            g2=random.uniform(N[1],N[2])
            g3=random.uniform(N[3],1)
            ghz1=(1-g1)*I/(2**self.n)+g1*(np.dot(ghz_3,(ghz_3.conj().T)))
            ghz2=(1-g2)*I/(2**self.n)+g2*(np.dot(ghz_3,(ghz_3.conj().T)))
            ghz3=(1-g3)*I/(2**self.n)+g3*(np.dot(ghz_3,(ghz_3.conj().T)))
            self.Ghz1=np.concatenate((self.Ghz1,ghz1.reshape(1,2**self.n,2**self.n)), axis=0)
            self.Ghz2=np.concatenate((self.Ghz2,ghz2.reshape(1,2**self.n,2**self.n)), axis=0)
            self.Ghz3=np.concatenate((self.Ghz3,ghz3.reshape(1,2**self.n,2**self.n)), axis=0) 
        la1=np.zeros((len(self.Ghz1)-1,self.n))
        la1[:,0]=1
        la2=np.zeros((len(self.Ghz2)-1,self.n))
        la2[:,1]=1
        la3=np.zeros((len(self.Ghz3)-1,self.n))
        la3[:,2]=1
        GHZ=np.concatenate((self.Ghz1[1:],self.Ghz2[1:],self.Ghz3[1:]), axis=0)
        la=np.concatenate((la1,la2,la3), axis=0)
        permutation= list(np.random.permutation(len(GHZ)))
        self.GHZ=GHZ[permutation]
        self.la=la[permutation]
        if v==0:
            return self.GHZ,self.la
        else:
            return self.Ghz1[1:],la1, self.Ghz2[1:],la2, self.Ghz3[1:],la3  #test data 
    def Un_State(self,la_data,lA):
        UnS=np.zeros((2**self.n,2**self.n)).reshape(1,2**self.n,2**self.n)
        ghz_3=((1/np.sqrt(2))*(I[0,:]+I[2**self.n-1,:])).reshape(2**self.n,1) 
        for i in range(self.number2):
            g1=random.uniform(0,1)
            ghz1=(1-g1)*I/8+g1*(np.dot(ghz_3,(ghz_3.conj().T)))
            UnS=np.concatenate((UnS,ghz1.reshape(1,2**self.n,2**self.n)), axis=0)
        itemindex_s1 = np.argwhere(lA[:,0]==1) #定位全可分
        se1 = (la_data[itemindex_s1]).reshape(len(itemindex_s1),2**self.n,2**self.n)
        r1=int(0.070*self.number2/len(se1))
        sea1=Augmentation_Strategies(data=se1,K=r1,n=self.n).Augment(A=1)
        #sec1=Convex_Combination(sea1,1)
        itemindex_s2 = np.argwhere((lA[:,1]==1))#定位可分
        se2 = la_data[itemindex_s2].reshape(len(itemindex_s2),2**self.n,2**self.n)
        r2=int(0.17*self.number2/len(se2))
        sea2=Augmentation_Strategies(data=se2,K=r2,n=self.n).Augment(A=1)
        #sec2=Convex_Combination(sea2,1)
        UnS=UnS[1:,:,:]
        UnD=np.concatenate((sea1,sea2,UnS), axis=0)
        permutation=list(np.random.permutation(len(UnS)))
        UnD=UnD[permutation]
        return UnD[0:self.number2]
    def test_Ghz(self,data1,data2,data3):
        augment=Augmentation_Strategies(K=1,n=self.n)
        Gh1=Augmentation_Strategies(data=data1,K=1,n=self.n).Augment(A=1)
        Gh2=Augmentation_Strategies(data=data2,K=1,n=self.n).Augment(A=1)
        Gh3=Augmentation_Strategies(data=data3,K=1,n=self.n).Augment(A=1)
        return Gh1,Gh2,Gh3
    
    
    
    
    
    
    
    
    
    
    
    
    