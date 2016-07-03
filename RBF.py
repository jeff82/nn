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
        self.beta = random.random((self.numCenters,1))
        self.W = random.random((self.numCenters, self.outdim))*2-1
        self.ERR=[]
        self.allE=[]
        self.cost=0

    def _basisfunc(self,c, d,cid):
#        print c,"ccc"
#        print d,"xxx",c-d
      #  print exp(-0.5/(self.beta[cid]*self.beta[cid]) * norm(c.T-d.T)**2),norm(c-d)**2,norm(c.T-d.T)**2,len(d),self.indim
<<<<<<< HEAD
        assert len(d) == self.indim
        print c
        print d
        return exp(-0.5/(self.beta[cid]*self.beta[cid]) * norm(c-d)**2)
=======
        #if len(d) != self.indim:
         #   print d,"d",self.indim,len(d)
        return exp(-0.5/(self.beta[cid]**2) * norm(c-d)**2)
>>>>>>> origin/master
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


    def gradindown(self,x,y,mode="instantaneous"):
        delta_Centre    = zeros((self.numCenters,self.indim),float)
        delta_Beta      = zeros((self.numCenters,self.indim),float)
        delta_Weight    = zeros((self.numCenters,self.indim),float)
        output          = zeros((self.outdim,1),float)
        err             = zeros((self.outdim,1),float)
        

        matCntr=[]
        mat2=[]
<<<<<<< HEAD
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
        

=======
        self.cost=0
        #print s3.size
        if mode == "instantaneous":
            for idd, xi in enumerate(x):
                G=self._calcAct(array(xi))
                output=dot(G, self.W)
                
                matCntr=xi-self.centers   
                err=-(y[idd]-output)
                s3=self.W*(err*G.reshape(self.numCenters,1))
                self.cost+=norm(err)**2
               # self._calcErr(y,output)                    
                mat2=array(map(sum,matCntr**2)).reshape(self.numCenters,1)
                delta_Centre+=s3*matCntr/(self.beta**2)
                delta_Beta+=s3*mat2/(self.beta**3)
                delta_Weight+=s3.reshape(self.numCenters,1)
            self.W-=eta*delta_Weight/float(self.indim)
            self.centers-=eta*delta_Centre/float(self.indim)
            self.beta-=eta*delta_Beta/float(self.indim)
           # print self.cost
            
        elif mode == "overall":
            for idd, xi in enumerate(x):
                G=self._calcAct(array(xi))
                output=dot(G, self.W)
                
                matCntr=xi-self.centers   
                err=-(y[idd]-output)
                s3=self.W*(err*G.reshape(self.numCenters,1))
                self.cost+=norm(err)**2
               # self._calcErr(y,output)                    
                mat2=array(map(sum,matCntr**2)).reshape(self.numCenters,1)
                delta_Centre=s3*matCntr/(self.beta**2)
                delta_Beta  =s3*mat2/(self.beta**3)
                delta_Weight=s3.reshape(self.numCenters,1)
                self.W-=eta*delta_Weight/float(self.indim)
                self.centers-=eta*delta_Centre/float(self.indim)
                self.beta-=eta*delta_Beta/float(self.indim)
   
      
>>>>>>> origin/master

    def test(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

if __name__ == '__main__':
#----- 1D Example ------------------------------------------------
    n = 100
    ncenters=30
    indim=1
    outdim=1

    x = mgrid[0:1:complex(0,indim*n)].reshape(n, indim)
<<<<<<< HEAD

    y = array(sin(3*(x+0.5)**3 - 1))
=======
    #x=numpy.ndarray.tolist(x)*indim
    #x=array(x).reshape(indim,n).T
    print x
    # set y and add random noise
    y = array(sin(3*(x+0.5)**3 - 1))#+x[:,1]).reshape(n,outdim) 
   # y=2*x
    print "yy",y
    # y += random.normal(0, 0.1, y.shape)
    #  rbf regression
>>>>>>> origin/master
    rbf = RBF(indim, ncenters, outdim)
    rbf.beta=mgrid[0.2:0.2:complex(0,ncenters)].reshape(ncenters, 1)
    rnd_idx = random.permutation(x.shape[0])
    rbf.centers = array([x[random.randint(0,x.shape[0],1)[0]] for i in range(ncenters)])
    print rbf.W
    
    
    rbf.train(x, y)
<<<<<<< HEAD
    z = rbf.test(x)
    rbf._calcErr(y,z)
    for i in range(200):
        z2 = rbf.test(x)
        rbf._calcErr(y,z2)
        rbf.gradindown(x)
=======
#    z = rbf.test(x)
#    z2 = rbf.test(x)
#    rbf._calcErr(y,z2)
    for i in range(0):
        #rbf._calcErr(y,z2)
        rbf.gradindown(x,y,"overall")
        print rbf.cost
>>>>>>> origin/master
    z2 = rbf.test(x)

    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')

    # plot learned model
    #plt.plot(x, z, 'r-', linewidth=2)
    plt.plot(x, z2, 'g-', linewidth=2)
    plt.plot(x,z2-y)
    print norm(z2-y)**2,(norm(z2-y)**2)**0.5

    # plot rbfs
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')

#    for c in rbf.centers:
#     # RF prediction lines
#         cx = arange(c-0.7, c+0.7, 0.01)
#         cy = [rbf._basisfunc(array([cx_]), array([c]),array([ci]) for ci, cx_ in enumerate(cx)]
#         plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
#         plt.xlim(-1.2, 1.2)
    plt.show()