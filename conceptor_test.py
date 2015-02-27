"""
@author: Yuhuang Hu
@note: some general test cases
"""

import numpy as np;
import matplotlib.pyplot as pplot;
import conceptors.util as util;
from conceptors.net import ConceptorNetwork;

# Create conceptor network
net=ConceptorNetwork(200);

# Prepare testing data
p1=np.asarray(xrange(2000));
p1=np.sin(2*np.pi*p1/np.sqrt(75));
p2=np.asarray(xrange(2000));
p2=0.5*np.sin(2*np.pi*p2/np.sqrt(20))+np.sin(2*np.pi*p2/np.sqrt(40));
p=np.hstack((p1[None].T, p2[None].T));

# training
net.train(p);

# test readout
print "NRMSE readout: %f" % (util.nrmse(net.W_out.dot(net.all_train_args), net.all_train_outs));
print "mean NEMSE W: %f" % util.nrmse(net.W.dot(net.all_train_old_args), net.W_targets);

y=net.W_out.dot(net.all_train_args)

pplot.figure(1);
pplot.plot(xrange(1000), p[500:1500,1]);
pplot.plot(xrange(1000), y[0,1000:2000]);
pplot.title("Redout")
pplot.show();

# test conceptors

parameter_nl=0.1;
c_test_length=200;
state_nl=0.5;
W_noisy=net.W+parameter_nl*np.abs(net.W).dot(np.random.rand(net.num_neuron, net.num_neuron)-0.5);

x_ctestpl=np.zeros((5,c_test_length,net.num_pattern));
p_ctestpl=np.zeros((1,c_test_length,net.num_pattern));

for p in xrange(net.num_pattern):
  c=net.Cs[0][p];
  x=0.5*np.random.rand(net.num_neuron,1);
  
  for n in xrange(c_test_length):
    x=np.tanh(W_noisy.dot(x)+net.W_bias)+state_nl*(np.random.rand(net.num_neuron,1)-0.5);
    x=c.dot(x);
    x_ctestpl[:,n,p]=x[0:5,0];
    p_ctestpl[:,n,p]=net.W_out.dot(x);
    












