"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: test case for Autoconceptor Network
"""

import numpy as np;
import matplotlib.pyplot as pplot;
from conceptors.net import Autoconceptor;
import conceptors.util as util;


# create network

net=Autoconceptor(1, 200);

# Prepare testing data
p1=np.asarray(xrange(2000));
p1=np.sin(2*np.pi*p1/np.sqrt(75));
p2=np.asarray(xrange(2000));
#p2=0.5*np.sin(2*np.pi*p2/np.sqrt(20))+np.sin(2*np.pi*p2/np.sqrt(40));
p2=np.sin(2*np.pi*p2/np.sqrt(40));
ps=np.vstack((p1[None], p2[None]));

p=[];
#p.append(ps);
p.append(p1[None]);
p.append(p2[None]);

net.load(p, load_mode="complete");

print util.nrmse(net.all_train_dt_args, net.D.dot(net.all_train_old_args))

c1, x1=net.cue_conceptor(p1[None]);
c2, x2=net.cue_conceptor(p2[None]);

#c1=net.recall_conceptor(c1, x1);
#c2=net.recall_conceptor(c2, x2);

print "Autoconceptors are trained"

measure_washout=50;
measure_rl=500;
x=x1;
x_before=x;

y1=np.zeros((measure_rl,1));
for n in xrange(measure_washout):
  x=c1.dot(np.tanh(net.W.dot(x)+net.W_in.dot(p1[n+130])+net.bias));
  
for n in xrange(measure_rl):
  r=np.tanh(net.W.dot(x)+net.W_in.dot(p1[n+180])+net.bias)
  x=c1.dot(r);
  y1[n,:]=net.W_out.dot(r);
  
pplot.figure(3);
pplot.plot(xrange(measure_rl), y1);
pplot.show();
  
x=x_before;

c1, x=net.recall_conceptor(c1, x);

x_before=x;

y2=np.zeros((measure_rl,1));
for n in xrange(measure_washout):
  x=c1.dot(np.tanh(net.W.dot(x)+net.D.dot(x)+net.bias));
  
for n in xrange(measure_rl):
  r=np.tanh(net.W.dot(x)+net.D.dot(x)+net.bias)
  x=c1.dot(r);
  y2[n,:]=net.W_out.dot(r);
  
x=x_before;

### test plot

this_driver_int=net.p_templates[:,0];
this_out_int=y1;

L=this_out_int.size;
M=this_driver_int.size;

phase_matches=np.zeros((1,L-M));

for phase_shift in xrange(L-M):
  phase_matches[0, phase_shift]=np.linalg.norm(this_driver_int-this_out_int[phase_shift:phase_shift+M]);
  
maxInd=np.argmin(phase_matches);

p_aligned=this_out_int[maxInd+1:maxInd+M+1];

print this_driver_int.shape
print p_aligned.shape

pplot.figure(1);
pplot.plot(xrange(20), this_driver_int);
pplot.plot(xrange(20), p_aligned);
pplot.show();
