# coding: utf-8
from scipy import *
from scipy.linalg import norm, pinv
import numpy

from matplotlib import pyplot as plt

eta=0.01

class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = random.random((numCenters,indim))
        #numpy.array([random.uniform(2, 2, 2) for i in xrange(numCenters)]).reshape(indim,numCenters)
        self.beta = random.random((1,self.numCenters))
        self.W = random.random((self.numCenters, self.outdim))
        self.ERR=[]
        self.allE=[]

    def _basisfunc(self,c, d,cid):
#        print c,"ccc"
#        print d,"xxx",c-d
      #  print exp(-0.5/(self.beta[cid]*self.beta[cid]) * norm(c.T-d.T)**2),norm(c-d)**2,norm(c.T-d.T)**2,len(d),self.indim
        assert len(d) == self.indim
        print c
        print d
        return exp(-0.5/(self.beta[cid]*self.beta[cid]) * norm(c-d)**2)
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

        #print "center", self.centers
    # calculate activations of RBFs
        G = self._calcAct(X)
#        print "gg",G.shape  # calculate output weights (pseudoinverse)
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
        delta_Centre    = zeros((self.numCenters,self.indim),float)
        delta_Beta      = zeros((self.numCenters,self.indim),float)
        delta_Weight    = zeros((self.numCenters,self.indim),float)

        sum1=sum2=sum3=0
        matCntr=[]
        mat2=[]
        s3=self.ERR*G
        #print s3.size
        for idd, xi in enumerate(x):
            matCntr=xi-self.centers
            mat2=array(map(sum,matCntr**2)).reshape(self.numCenters,1)
            s1=s3[idd].reshape(self.numCenters,1)*matCntr
            s2=s3[idd].reshape(self.numCenters,1)*mat2
            sum1+=self.W*s1/(self.beta*self.beta)
            sum2+=self.W*s2/(self.beta**3)
            sum3+=s3[idd].reshape(self.numCenters,1)
            delta_Weight=sum3.reshape(self.numCenters,1)

        self.W-=eta*delta_Weight/float(x.shape[0])
        self.centers-=eta*sum1/float(x.shape[0])
#        print self.beta
        self.beta-=eta*sum2/float(x.shape[0])
        


    def test(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

if __name__ == '__main__':
#----- 1D Example ------------------------------------------------
    n = 25
    ncenters=5
    indim=1
    outdim=1

    x = mgrid[0:1:complex(0,indim*n)].reshape(n, indim)

    y = array(sin(3*(x+0.5)**3 - 1))
    rbf = RBF(indim, ncenters, outdim)
    rbf.beta=mgrid[0.2:0.2:complex(0,ncenters)].reshape(ncenters, 1)
    rnd_idx = random.permutation(x.shape[0])[:rbf.numCenters]
    rbf.centers = array([x[i,:] for i in rnd_idx])
    
    
    rbf.train(x, y)
    z = rbf.test(x)
    rbf._calcErr(y,z)
    for i in range(200):
        z2 = rbf.test(x)
        rbf._calcErr(y,z2)
        rbf.gradindown(x)
    z2 = rbf.test(x)

    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')

    # plot learned model
    plt.plot(x, z, 'r-', linewidth=2)
    plt.plot(x, z2, 'g-', linewidth=2)
    plt.plot(x,z-y)

    # plot rbfs
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')

#    for c in rbf.centers:
#     # RF prediction lines
#         cx = arange(c-0.7, c+0.7, 0.01)
#         cy = [rbf._basisfunc(array([cx_]), array([c]),array([ci]) for ci, cx_ in enumerate(cx)]
#         plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
#         plt.xlim(-1.2, 1.2)
    plt.show()