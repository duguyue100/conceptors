"""
@author: Yuhuang Hu
@note: some general test case
"""

from conceptors.util import gen_internal_weights;
from conceptors.util import init_weights;

#w=gen_internal_weights(5, 0.1);

#print w;

wstar, win, wbias=init_weights(30, 1.5, 1.5, 0.2);

print wstar, win, wbias;