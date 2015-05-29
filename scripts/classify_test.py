"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com
"""

import numpy as np;
import matplotlib.pyplot as plt;
import conceptors.util as util;
from conceptors.net import ConceptorNetwork;

Xtr, Ytr, Xte, Yte=util.load_CIFAR10("/home/arlmaster/workspace/conceptors/conceptors/data/CIFAR10");

Xtr=np.mean(Xtr, 3);
Xte=np.mean(Xte, 3);
Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])/255;
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])/255;
Xtrain=np.hstack((Ytr[None].T, Xtrain))[0:10000];
Xtrain=Xtrain[Xtrain[:,0].argsort()];

print "Dataset is created";

X=[];
num_train=Xtrain.shape[0];
idx=0;

for i in xrange(10):
  
  x_temp=np.asarray([]);
  while idx!=num_train and Xtrain[idx,0]==i:
    if not x_temp.size:
      x_temp=Xtrain[idx,1:];
    else:
      x_temp=np.vstack((x_temp, Xtrain[idx,1:]));
      
    idx+=1;
    
  print "Class %i is generated" % i;
    
  X.append(x_temp.T);


n_in=X[0].shape[0];
num_neuron=500;
net=ConceptorNetwork(num_in=n_in,
                     num_neuron=num_neuron,
                     washout_length=200,
                     learn_length=700);
                     
print "the network is created";

net.train(X);

print "the network is trained";

conceptors=net.Cs[0];

img=Xtest[1];

pos_evidence=np.zeros((1, len(conceptors)));
for i in xrange(len(conceptors)):
  
  # get activation
  
  x=np.zeros((net.num_neuron,1));
  for j in xrange(200):
    x=np.tanh(net.W_star.dot(x)+net.W_in.dot(img[None].T)+net.W_bias);
  
  pos_evidence[0,i]=x.T.dot(conceptors[i]).dot(x);
  
print np.argmax(pos_evidence);
print Yte[1];
  
  
  
  
  
  
  
  
  
  
  