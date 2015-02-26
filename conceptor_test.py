"""
@author: Yuhuang Hu
@note: some general test case
"""

import numpy as np;
import matplotlib.pyplot as pplot;
from conceptors.net import ConceptorNetwork;

net=ConceptorNetwork(200);

p1=np.asarray(xrange(2000));
p1=np.sin(2*np.pi*p1/10);
p2=np.asarray(xrange(2000));
p2=np.sin(2*np.pi*p2/15);
p=np.hstack((p1[None].T, p2[None].T));

net.train(p);
y1=net.recall(net.startXs[:,0]);
print y1.shape

y2=net.recall(net.startXs[:,1]);
print y2.shape

pplot.figure(1);
pplot.plot(xrange(200), p[0:200,1]);
pplot.plot(xrange(200), y2.T);
pplot.show();