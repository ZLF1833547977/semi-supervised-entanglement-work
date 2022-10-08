import numpy as np
import qutip as Q
from qutip import *
class Augmentation_Strategies():
    def __init__(self,data=None,K=2,n=2): #n-qubit state
        self.data=data
        self.K=K
        self.n=n
        self.ii=np.arange(0, self.n)
        self.Au=np.zeros((2**self.n,2**self.n)).reshape(1,2**self.n,2**self.n)
        self.Ub_hut=np.zeros((2**self.n,2**self.n)).reshape(1,2**self.n,2**self.n)
        if data is None:
            self.len=0
        else:
            self.len=len(data)
    def Unitary_trans(self,rho): 
        rho.reshape(2**self.n,2**self.n)
        for k in range(self.K):
            u={}
            para=np.random.randint(-10**8, 10**8, size=(self.n, 3))
            for (gama,beta,delta,i) in zip(para[:,0],para[:,1],para[:,2],self.ii):
                u["u{0}".format(i)]=np.exp(1j*np.random.randint(-10**8, 10**8))\
                    *np.dot(np.diag([np.exp(-1j*beta/2),np.exp(1j*beta/2)]),\
                            np.array([[np.cos(gama/2),-np.sin(gama/2)],[np.sin(gama/2),np.cos(gama/2)]]),\
                                np.diag([np.exp(-1j*delta/2),np.exp(1j*delta/2)]))        
                if i==0:
                   UT=Qobj(u['u0'])
                else:
                   UT=Q.tensor(UT,Qobj(u["u{0}".format(i)]))
            Rho_augment=UT*rho*(UT.conj().trans()) 
            self.Au=np.concatenate((self.Au,Rho_augment.reshape(1,2**self.n,2**self.n)), axis=0)
        self.Au=self.Au[1:,:,:]
        return self.Au
    def Convex_Combination(self,B=1):#
        permutation1 = list(np.random.permutation(self.len))
        se1 = self.data[permutation1]
        for i in range(B):
            for (x1,x2) in zip(se1,self.data):       
                bb=np.random.dirichlet(np.ones(2), size=1).reshape(2,1)
                x_hut=(bb[0]*x1+bb[1]*x2).reshape(1,2**self.n,2**self.n)
                self.Au=np.concatenate((self.Au,x_hut), axis=0) 
            se1=self.Au[1+i*self.len:(i+1)*self.len+1,:,:]
            permutation2= list(np.random.permutation(self.len))
            se1=se1[permutation2]
        self.Au =self.Au[1:,:,:] 
        return self.Au
    def Augment(self,A=0):
        for ub in self.data:
                ub_hut= Augmentation_Strategies(data=self.data,K=self.K,n=self.n).Unitary_trans(ub)
                if A==0:
                    self.Ub_hut=np.concatenate((self.Ub_hut,ub_hut,ub.reshape(1,2**self.n,2**self.n)), axis=0)
                    #remain the original sample
                else:
                    self.Ub_hut=np.concatenate((self.Ub_hut,ub_hut), axis=0)
                    #discard the original sample
        self.Ub_hut=self.Ub_hut[1:,:,:]
        return self.Ub_hut
    def Sharpen(self,qb_average,T=0.5):#qb_average is one-hot label 
        qb=np.zeros((1,self.n))  
        for pj in qb_average:  
            pJ=pj**(1/T)/(pj[0]**(1/T)+(pj[1])**(1/T))
            qb=np.concatenate((qb,pJ.reshape(1,self.n)), axis=0)
        qb=qb[1:,:]
        return qb 
    def Index(self,index):
        index1=np.zeros((1,))
        ii=np.ones((len(index1)))
        indexx1=(self.K+1)*index
        for k in range(self.K+1):
            indexk=indexx1+k*ii
            index1=np.concatenate((index1,indexk), axis=0)
        index1=np.sort(index1,axis=0)
        index1=index1.astype(np.int64)
        index1=index1[1:]
        return index1
    def QB(self,qb):
        #qb = to_categorical(qb, num_classes=None)
        Yb_hut=np.zeros(((self.K+1)*len(qb),self.n)) 
        for k in range(self.K+1):
            Yb_hut[np.arange(k,len(Yb_hut),self.K+1),:]=qb
        return Yb_hut
class Feature_Trans():
    def __init__(self,data=None):
        self.data=data
    def pauli_express(self):
        R=np.zeros((4,4)).reshape(1,4,4)
        for i in range(len(self.data)):
            Rho=self.data[i,:,:]
            sigma=[Q.identity(2),Q.sigmax(), Q.sigmay(), Q.sigmaz()]
            a=[]
            for j in range(0,4):
               for k in range(0,4):          
                   a.append(np.real(np.trace(np.dot(Rho,Q.tensor(sigma[j],sigma[k]))))) 
            a=np.array(a).reshape(1,4,4)
            R=np.concatenate((R,a), axis=0)
        R=R[1:,:,:]
        return R 
    def RM(self):
        return np.concatenate((self.data.real,self.data.imag), axis=1)
def Gauss_fun(x):
    y=np.exp(-x*x)/np. sqrt(2*np.pi)
    return y