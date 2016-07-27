# - coding: utf-8 -

import numpy
from sklearn.decomposition import RandomizedPCA
    
from PIL import Image
#from tools import make_tile
from pic import *
import gzip
import cPickle
import os


dir=os.getcwd()+r'\dcfls\fuck pic\subpic'

if platform.system() is 'Windows':
    dir=os.getcwd()+r'\dcfls\fuck pic\subpic'
    name='\grid1xx.jpg'
else:
   dir=os.getcwd()+r'/dcfls/fuck pic/subpic'
   name='/grid1xx.jpg'

def GausianF(beta,dist,centre=0):
    return exp(-0.5/(self.beta[cid]*self.beta[cid]) * norm(centre-dist)**2)
class SOM():
    def __init__(self, x, y):        
        self.map = []
        self.n_neurons = x*y
        self.sigma = x
        self.template = numpy.arange(x*y).reshape(self.n_neurons,1)
        self.alpha = 0.6
        self.alpha_final = 0.1
        self.shape = [x,y]
        self.epoch = 0
        
    def train(self, X, iter, batch_size=1):
        if len(self.map) == 0:
            x,y = self.shape
            # first we initialize the map
            self.map = numpy.zeros((self.n_neurons, len(X[0])))
            
            # then we the pricipal components of the input data
            eigen = RandomizedPCA(10).fit_transform(X.T).T
            
            # then we set different point on the map equal to principal components to force diversification
            self.map[0] = eigen[0]
            self.map[y-1] = eigen[1]
            self.map[(x-1)*y] = eigen[2]
            self.map[x*y - 1] = eigen[3]
            for i in range(4, 10):
                self.map[numpy.random.randint(1, self.n_neurons)] = eigen[i]
                
        self.total = iter
        
        # coefficient of decay for learning rate alpha
        self.alpha_decay = (self.alpha_final/self.alpha)**(1.0/self.total)
        
        # coefficient of decay for gaussian smoothing
        self.sigma_decay = (numpy.sqrt(self.shape[0])/(4*self.sigma))**(1.0/self.total)
        
        samples = numpy.arange(len(X))
        numpy.random.shuffle(samples)
    
        for i in xrange(iter):
            idx = samples[i:i + batch_size]
            self.iterate(X[idx])
    
    def transform(self, X):
        # We simply compute the dot product of the input with the transpose of the map to get the new input vectors
        res = numpy.dot(numpy.exp(X),numpy.exp(self.map.T))/numpy.sum(numpy.exp(self.map), axis=1)
        res = res / (numpy.exp(numpy.max(res)) + 1e-8)
        return res
     
    def iterate(self, vector):  
        x, y = self.shape
        
        delta = self.map - vector
        
        # Euclidian distance of each neurons with the example
        dists = numpy.sum((delta)**2, axis=1).reshape(x,y)
        
        # Best maching unit
        idx = numpy.argmin(dists)
       # print "Epoch ", self.epoch, ": ", (idx/x, idx%y), "; Sigma: ", self.sigma, "; alpha: ", self.alpha
        
        # Linearly reducing the width of Gaussian Kernel
        self.sigma = self.sigma*self.sigma_decay
        dist_map = self.template.reshape(x,y)     
        
        # Distance of each neurons in the map from the best matching neuron
        dists = numpy.sqrt((dist_map/x - idx/x)**2 + (numpy.mod(dist_map,x) - idx%y)**2).reshape(self.n_neurons, 1)
        #dists = self.template - idx
        
        # Applying Gaussian smoothing to distances of neurons from best matching neuron
        h = numpy.exp(-(dists/self.sigma)**2)      
         
        # Updating neurons in the map
        self.map -= self.alpha*h*delta
       
        # Decreasing alpha
        self.alpha = self.alpha*self.alpha_decay
        
        self.epoch = self.epoch + 1 
        
        
##################################################### 
#       EXAMPLE: TRAINING SOM ON MNIST DATA         #
#####################################################       


def load_mnist():   
    f = gzip.open("mnist/mnist.pkl.gz", 'rb')
    train, valid, test = cPickle.load(f)
    f.close()  
    return train[0][:20000],train[1][:20000]


def _split_(inimg, splitsz,discrete=10):
    sx,sy,color=inimg.shape
    print sx,sy
    sub=0
    sub=min(sx,sy)/discrete
    print sub
    spic=numpy.zeros((sub**2,splitsz,splitsz),dtype='uint8')
    
#    img=inimg[:,:,2]
#    x,y=img.shape
#    difx=numpy.zeros((sub**2,splitsz,splitsz),dtype='uint8')
#    dify=numpy.zeros((sub**2,splitsz,splitsz),dtype='uint8')
#    difx=img[:x-2,:]-img[2:x,:]
#    dify=img[:,:y-2]-img[:,2:y]
#    difx = Image.fromarray(difx)
#    dify  = Image.fromarray(dify)
#    difx.show()
#    dify.show()
        
    i=0
    #out_array = numpy.zeros(out_shape, dtype='uint8')
    for imw in numpy.arange(1,min(sx,sy),discrete):
        for imh in numpy.arange(1,min(sx,sy),discrete):
            try:
                spic[i,0:splitsz,0:splitsz]=inimg[imw:imw+splitsz,imh:imh+splitsz,2]
                img = Image.fromarray(spic[i ,:,:])
                #print img
                img.save(dir+"\som_results"+str(i)+".jpg")
                #print 'ok'
            except:
                pass
               # print imw,imh,splitsz
            
            i=i+1
            #print spic[imw*imh,:,:]
            #img = Image.fromarray(spic[imw*imh,:,:])
           # print img
            
            
            #print "NO.",imw*imh
        
    
def demo():
    # Get data
    #X, y = load_mnist()
    X,img=readimg()
    print X.shape
    
    _split_(X,32)
#    cl = SOM(20, 20)
#    cl.train(X, 2000)  
#    
#    # Plotting hidden units
#    W = cl.map
#    W = make_tile(W, img_shape= (28,28), tile_shape=(20,20))
#    img = Image.fromarray(W)
#    img.save("som_results.png")
#    
#    
#    # creating new inputs
#    X = cl.transform(X)
#   
#    # we can plot "landscape" 3d to view the map in 3d
#    landscape = cl.transform(numpy.ones((1,28**2)))
    return# cl.map, X, y,landscape
    
class Corner():
    def __init__(self, x, y,img): 
        self.windowsz=5
        self.gausainwidth=3
        self.X_w=numpy.array([1,0,-1])
        self.Y_w=numpy.array([1,0,-1])
        self.img=img
        self.difx=[]
        self.dify=[]
    def conv(self,dif,G):
        '''
        2D assumed
        '''
        dx,dy=dif.shape
        dxg,dyg=G.shape
        sum=0
        cv=0
        for x in range(self.windowsz):
            for y in range(self.windowsz):
                cv=numpy.dot(dif[x,y],G[x,y])
                sum+=cv
            
        return sum
        
    def difpic(self,img):
        x,y=img.shape
        difx=img[:x-2,:]-img[2:x,:]
        dify=img[:,:y-2]-img[:,2:y]
        self.difx=difx
        self.dify=dify
        return difx,dify
   
   def cvfpic(self,img):
       difx,dify = difpic(img)
       px,py=img.shape
       slipx=px-self.windowsz+1
       slipy=py-self.windowsz+1
       for x in range(slipx):
           for y in range(slipy):
               A=conv(numpy.array(difx**2),numpy.array(range(self.windowsz)))
               B=conv(numpy.array(dify**2),numpy.array(range(self.windowsz)))
               C=conv(numpy.array(difx*dify),numpy.array(range(self.windowsz)))
           pass
       return
                
        
    
if __name__ == '__main__':
    demo()