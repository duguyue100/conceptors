'''
Created on Feb 25, 2015

@author: Yuhuang Hu
@note: useful utilities functions for assisting conceptor networks
'''

import numpy as np;
import scipy.sparse.linalg;

def gen_internal_weights(num_neuron,
                         density):
  """
  Generate internal weights in a conceptor network
  
  @param num_neuron: number of neurons in the conceptor network
  @param density: density of the conceptor network (connectivity)
  
  @return: weights: a dense matrix of internal weights
  """
  
  weights=scipy.sparse.rand(m=num_neuron, n=num_neuron, density=density, format='coo');
  eigw, _=scipy.sparse.linalg.eigsh(weights, 1);
  weights/=np.abs(eigw[0]);
  
  return weights.toarray();

def init_weights(num_in,
                 num_neuron,
                 sr,
                 in_scale,
                 bias_scale):
  """
  Initialize weights for a new conceptor network
  
  @param num_in: number of input
  @param num_neuron: number of internal neurons 
  @param sr: spectral radius
  @param in_scale: scaling of pattern feeding weights
  @param bias_scale: size of bias
  """
  
  # generate internal weights
  if num_neuron<=20:
    W_star_raw=gen_internal_weights(num_neuron=num_neuron, density=1);
  else:
    W_star_raw=gen_internal_weights(num_neuron=num_neuron, density=10./num_neuron);
    
  W_star=W_star_raw*sr;
  
  # generate input weights
  W_in=np.random.rand(num_neuron, num_in)*in_scale;
  
  # generate bias
  W_bias=np.random.rand(num_neuron,1)*bias_scale;
    
  return W_star, W_in, W_bias;

def nrmse(output,
          target):
  """
  Compute normalized root mean square error.
    
  @param output: output data in D x time dimension
  @param target: target data in D x time dimension
    
  @return NRMSE: normalized root mean square error. 
  """
    
  if output.ndim==1 and target.ndim==1:
    output=output[None].T;
    target=target[None].T;
    
  combined_var=0.5*(np.var(a=target, axis=1, ddof=1)+np.var(a=output, axis=1, ddof=1));    
  error_signal=(output-target);
    
  return np.mean(np.sqrt(np.mean(error_signal**2, 1)/combined_var));








