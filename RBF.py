# coding: utf-8
from scipy import *
from scipy.linalg import norm, pinv
import numpy

from matplotlib import pyplot as plt

eta=0.001

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
        print c,"ccc"
        print d,"xxx",c-d
        print exp(-0.5/(self.beta[cid]*self.beta[cid]) * norm(c.T-d.T)**2),norm(c-d)**2,norm(c.T-d.T)**2,len(d),self.indim
        assert len(d) == self.indim
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
        print "dddd"
        matCntr=numpy.ndarray.tolist(self.centers)
        print self.centers
        matCntr=array(matCntr*G.shape[0]).reshape(G.shape[0],self.numCenters)
        print matCntr
        print x
        Dist=x-matCntr
        s3=(self.ERR*G)
        s1=s3*Dist
        s2=s1*Dist        
        print s1
        
#        delta_center=eta*Weight[j]/(delta[j]*delta[j])*sum1;
#        delta_delta=eta*Weight[j]/pow(delta[j],3)*sum2;
#        delta_weight=eta*sum3;
#        center[j]+=delta_center;
#        delta[j]+=delta_delta;
#        Weight[j]+=delta_weight;
#             sum1+=self.ERR[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j]);
#             sum2+=self.ERR[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j])*(X[i]-center[j]);
#             sum3+=self.ERR[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]));
        

    def test(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

if __name__ == '__main__':
#----- 1D Example ------------------------------------------------
    n = 4
    ncenters=3
    indim=2

    x = mgrid[-1:1:complex(0,n)].reshape(n, 1)
    x=numpy.ndarray.tolist(x)*indim
    x=array(x).reshape(indim,n).T
    print x
    # set y and add random noise
    y = sin(3*(x+0.5)**3 - 1)
    # y += random.normal(0, 0.1, y.shape)
    #  rbf regression
    rbf = RBF(indim, ncenters, 1)
    rbf.beta=mgrid[0.2:0.2:complex(0,ncenters)].reshape(ncenters, 1)
    rnd_idx = random.permutation(x.shape[0])[:rbf.numCenters]
    rbf.centers = array([x[i,:] for i in rnd_idx])
    
    
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

#    for c in rbf.centers:
#     # RF prediction lines
#         cx = arange(c-0.7, c+0.7, 0.01)
#         cy = [rbf._basisfunc(array([cx_]), array([c]),array([ci]) for ci, cx_ in enumerate(cx)]
#         plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
#         plt.xlim(-1.2, 1.2)
    plt.show()