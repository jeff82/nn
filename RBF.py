# coding: utf-8
from scipy import *
from scipy.linalg import norm, pinv
import numpy

from matplotlib import pyplot as plt


class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = []
        self.W = random.random((self.numCenters, self.outdim))
        self.ERR=[]
        self.allE=[]

    def _basisfunc(self,c, d,cid):
        assert len(d) == self.indim
        return exp(-0.5/self.beta[cid] * norm(c-d)**2)
    def _calcAct(self, X):
    #calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x,ci)
        return G
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        print "rnd_idx", rnd_idx.shape
        self.centers = [X[i,:] for i in rnd_idx]
        #print "center", self.centers
    # calculate activations of RBFs
        G = self._calcAct(X)
        print len(G)  # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)
        #self.ERR=zeros((X.shape[0], self.outdim), float)
        self.allE =zeros((1,self.outdim),float)
    def _calcErr(self,y,z):
        self.ERR=y-z
        self.allE = (norm(self.ERR))
        #print "sigle err",self.ERR
        #print "all err",self.allE


    def gradindown(self,x):
        G = self._calcAct(x)
        delta_Centre    = zeros((self.indim,self.numCenters),float)
        delta_Beta      = zeros((self.indim,self.numCenters),float)
        delta_Weight    = zeros((self.indim,self.numCenters),float)

        sum1=sum2=sum3=0
        
        Dist=array([x-float(cs) for i,cs in enumerate(self.centers)])
        print  "dist",Dist[0], Dist.shape,x.shape
        bt=[[float(sb)] for j,sb in enumerate(self.beta)]
        bt=array(bt)*array(bt)
        #print "bt",bt
        s1=(self.ERR*Dist)
        print s1,s1.shape,self.ERR
#         for i in range(len(G))
#             sum1+=self.ERR[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j]);
#             sum2+=self.ERR[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j])*(X[i]-center[j]);
#             sum3+=self.ERR[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]));
        

    def test(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

if __name__ == '__main__':
#----- 1D Example ------------------------------------------------
    n = 12

    x = mgrid[-1:1:complex(0,n)].reshape(n, 1)
    print x.shape
    # set y and add random noise
    y = sin(3*(x+0.5)**3 - 1)
    # y += random.normal(0, 0.1, y.shape)
    #  rbf regression
    rbf = RBF(1, 10, 1)
    rbf.beta=mgrid[0.2:0.2:complex(0,10)].reshape(10, 1)
    rbf.train(x, y)
    z = rbf.test(x)
    rbf._calcErr(y,z)
    rbf.gradindown(x)

    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')

    # plot learned model
    plt.plot(x, z, 'r-', linewidth=2)
    plt.plot(x,z-y)

    # plot rbfs
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')

    #for c in rbf.centers:
    # # RF prediction lines
    #     cx = arange(c-0.7, c+0.7, 0.01)
    #     cy = [rbf._basisfunc(array([cx_]), array([c]),array([ci]) for ci, cx_ in enumerate(cx)]
    #     plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
    #     plt.xlim(-1.2, 1.2)
    plt.show()
