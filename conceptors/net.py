'''
Created on Feb 25, 2015

@author: Yuhuang Hu
@note: a concept network implementation
'''

import numpy as np;
import conceptors.util;

class ConceptorNetwork:
  """
  A implementaion of conceptor network
  """
  
  def __init__(self,
               num_neuron,
               num_pattern,
               sr=1.5,
               in_scale=1.5,
               bias_scale=0.2,
               washout_length=500,
               learn_length=1000):
    
    """
    Initialize conceptor network
    
    @param num_neuron: number of internal neurons  
    @param sr: spectral radius
    @param in_scale: scaling of pattern feeding weights
    @param bias_scale: size of bias 
    @param num_pattern: number of pattern
    @param washout_length: length of wash-out iteration
    @param learn_length: length of learning iteration
    """
    
    # document parameters
    self.num_neuron=num_neuron;
    self.num_pattern=num_pattern;
    self.sr=sr;
    self.in_scale=in_scale;
    self.bias_scale=bias_scale;
    self.learn_length=learn_length;
    self.washout_length=washout_length;
    
    # initialize weights and parameters
    W_star, W_in, W_bias=conceptors.util.init_weights(num_neuron,
                                                      sr,
                                                      in_scale,
                                                      bias_scale);
                                                                     
    self.W_star=W_star;
    self.W_in=W_in;
    self.W_bias=W_bias;
                                                                     
    self.all_train_args=np.zeros(num_neuron, num_pattern*learn_length);
    self.all_train_old_args=np.zeros(num_neuron, num_pattern*learn_length);
    self.all_train_targs=np.zeros(num_neuron, num_pattern*learn_length);
    self.all_train_outs=np.zeros(1,num_pattern*learn_length);
    
    # initialize collectors
    
    self.readout_weights=[];
    self.pattern_collectors=[];
    self.x_collectors_centered=[];
    self.x_collectors=[];
    self.sr_collectors=[];
    self.ur_collectors=[];
    self.pattern_rs=[];
    self.train_xpl=[];
    self.train_ppl=[];
    
    self.startXs=np.zeros(num_neuron, num_pattern);
    
    
  def train_pattern(self,
                    pattern):
    """
    This function train one single input pattern.
    
    @param pattern: input pattern
    """
    
    x_collector=np.zeros(self.num_neuron, self.learn_length);
    x_old_collector=np.zeros(self.num_neuron, self.learn_length);
    p_collector=np.zeros(1, self.learn_length);
    x=np.zeros(self.num_neuron, 1);
    
    for n in xrange(self.washout_length+self.learn_length):
      u=pattern[n];
      x_old=x;
      x=np.tanh(self.W_star.dot(x)+self.W_in.dot(x)+self.W_bias);
      if n>self.washout_length-1:
        x_collector[:, n-self.washout_length]=x;
        x_old_collector[:, n-self.washout_length]=x_old;
        p_collector[0, n-self.washout_length]=u;
    
  
  def train(self):
    
    pass
  
    