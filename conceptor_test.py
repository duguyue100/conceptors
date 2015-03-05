"""
@author: Yuhuang Hu
@note: some general test cases
"""

import numpy as np;
import scipy.interpolate;
import matplotlib.pyplot as pplot;
import conceptors.util as util;
from conceptors.net import ConceptorNetwork;
from conceptors.util import read_jpv_data;
from conceptors.util import normalize_jap_data;
from conceptors.util import transform_jap_data;

train_inputs, train_outputs, test_inputs, test_outputs=read_jpv_data("/home/arlmaster/workspace/conceptors/conceptors/data/ae.train",
                                                                     "/home/arlmaster/workspace/conceptors/conceptors/data/ae.test");
                                                                     
                                                                     
train_data, shifts, scales=normalize_jap_data(train_inputs);
test_data=transform_jap_data(test_inputs, shifts, scales);



# Create conceptor network
net=ConceptorNetwork(2, 200);

# Prepare testing data
p1=np.asarray(xrange(2000));
p1=np.sin(2*np.pi*p1/np.sqrt(75));
p2=np.asarray(xrange(2000));
#p2=0.5*np.sin(2*np.pi*p2/np.sqrt(20))+np.sin(2*np.pi*p2/np.sqrt(40));
p2=np.sin(2*np.pi*p2/np.sqrt(40));
ps=np.vstack((p1[None], p2[None]));

p=[];
p.append(ps);
#p.append(p1[None]);
#p.append(p2[None]);

# training
net.train(p);

# test readout
print "NRMSE readout: %f" % (util.nrmse(net.W_out.dot(net.all_train_args), net.all_train_outs));
print "mean NEMSE W: %f" % util.nrmse(net.W.dot(net.all_train_old_args), net.W_targets);

y=net.W_out.dot(net.all_train_args)
print y.shape
print net.all_train_outs.shape

pplot.figure(1);
pplot.plot(xrange(1000), p[0][0,500:1500]);
pplot.plot(xrange(1000), y[0,0:1000]);
pplot.title("Redout")
pplot.show();

# test conceptors

parameter_nl=0.1;
c_test_length=200;
state_nl=0.5;
W_noisy=net.W+parameter_nl*np.abs(net.W).dot(np.random.rand(net.num_neuron, net.num_neuron)-0.5);

x_ctestpl=np.zeros((5,c_test_length,net.num_pattern));
p_ctestpl=np.zeros((net.num_in,c_test_length,net.num_pattern));

for p in xrange(net.num_pattern):
  c=net.Cs[0][p];
  x=0.5*np.random.rand(net.num_neuron,1);
  
  for n in xrange(c_test_length):
    x=np.tanh(W_noisy.dot(x)+net.W_bias)+state_nl*(np.random.rand(net.num_neuron,1)-0.5);
    x=c.dot(x);
    x_ctestpl[:,n,p]=x[0:5,0];
    p_ctestpl[:,n,p]=net.W_out.dot(x)[:,0];
    
    
for p in xrange(net.num_pattern):
  int_rate=20;
  this_driver=net.train_ppl[p];
  this_out=p_ctestpl[0,:,p];
  this_driver_int_f=scipy.interpolate.interp1d(np.arange(0,net.signal_plot_length), this_driver, kind="cubic");
  this_driver_int=this_driver_int_f(np.linspace(0, net.signal_plot_length-1, int_rate*net.signal_plot_length));
  this_out_int_f=scipy.interpolate.interp1d(np.arange(0,c_test_length), this_out, kind="cubic");
  this_out_int=this_out_int_f(np.linspace(0, c_test_length-1, int_rate*c_test_length));
  
      
pplot.figure(2);
pplot.plot(np.log10(net.sr_collectors[0]))
pplot.plot(net.Cs[2][0])
#pplot.plot(xrange(c_test_length), p_ctestpl[0,:,1]);
pplot.show()

# draw conceptors

L=100;

trace1=net.all_train_args[[71,80],0:L];
trace2=net.all_train_args[[71,80],1000:1000+L];

R1=trace1.dot(trace1.T)/L;
U1, S1, _=np.linalg.svd(R1);
R2=trace2.dot(trace2.T)/L;
U2, S2, _=np.linalg.svd(R2);

cycle_data=np.vstack((np.cos(2*np.pi*np.arange(0,200)/200), np.sin(2*np.pi*np.arange(0,200)/200)));

E1=R1.dot(cycle_data); E2=R2.dot(cycle_data);

a=1.6
C1=R1.dot(np.linalg.inv(R1+a**-2*np.eye(2)));
U1c, S1c, _=np.linalg.svd(C1);
C2=R2.dot(np.linalg.inv(R2+a**-2*np.eye(2)));
U2c, S2c, _=np.linalg.svd(C2);
E1c=C1.dot(cycle_data);
E2c=C2.dot(cycle_data);

pplot.figure(3);
pplot.plot(cycle_data[0,:], cycle_data[1,:]);
#pplot.plot(trace1[0,:], trace1[1,:]);
pplot.plot(E1c[0,:], E1c[1,:]);
pplot.show();






