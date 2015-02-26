"""
@author: Yuhuang Hu
@note: some general test case
"""

import numpy as np;
from conceptors.util import gen_internal_weights;
from conceptors.util import init_weights;
from conceptors.util import nrmse;

#w=gen_internal_weights(5, 0.1);

#print w;

wstar, win, wbias=init_weights(30, 1.5, 1.5, 0.2);

#print wstar, win, wbias;

print nrmse(np.asarray([[1,2,3,4]]), np.asarray([[4,5,6,7]]))