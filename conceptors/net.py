'''
Created on Feb 25, 2015

@author: Yuhuang Hu
@note: a concept network implementation
'''

import numpy as np;
import conceptors.util;
import numpy.matlib;

class ConceptorNetwork:
  """
  A implementaion of conceptor network
  
  This class is designed by following:
  Section 3.2 Driving a Reservorir with Different Patterns in
  Controlling Recurrent Neural Networks by Conceptors
  """
  
  def __init__(self,
               num_neuron,
               sr=1.5,
               in_scale=1.5,
               bias_scale=0.2,
               washout_length=500,
               learn_length=1000,
               signal_plot_length=20,
               tychonov_alpha_readout=0.01):
    
    """
    Initialize conceptor network
    
    @param num_neuron: number of internal neurons  
    @param sr: spectral radius
    @param in_scale: scaling of pattern feeding weights
    @param bias_scale: size of bias 
    @param washout_length: length of wash-out iteration
    @param learn_length: length of learning iteration
    @param signal_plot_length: length of plot length
    @param tychnonv_alpha_readout: Tychonov regularization parameter
    """
    
    # document parameters
    self.num_neuron=num_neuron;
    self.num_pattern=0;
    self.sr=sr;
    self.in_scale=in_scale;
    self.bias_scale=bias_scale;
    self.learn_length=learn_length;
    self.washout_length=washout_length;
    self.signal_plot_length=signal_plot_length;
    self.tychonov_alpha_readout=tychonov_alpha_readout;
    
    # initialize weights and parameters
    W_star, W_in, W_bias=conceptors.util.init_weights(num_neuron,
                                                      sr,
                                                      in_scale,
                                                      bias_scale);
                                                                     
    self.W_star=W_star;
    self.W_in=W_in;
    self.W_bias=W_bias;
    self.W_out=np.asarray([]);
    self.W_targets=np.asarray([]);
    self.W=np.asarray([]);
                                                                     
    self.all_train_args=np.asarray([]);
    self.all_train_old_args=np.asarray([]);
    self.all_train_targs=np.asarray([]);
    self.all_train_outs=np.asarray([]);
    
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
    
    self.startXs=np.asarray([]);
    
    self.Cs=[]; self.Cs.append([]); self.Cs.append([]); self.Cs.append([]); self.Cs.append([]);
    
    
  def train_pattern(self,
                    pattern):
    """
    This function train one single input pattern.
    
    @param pattern: input pattern
    """
    
    x_collector=np.zeros((self.num_neuron, self.learn_length));
    x_old_collector=np.zeros((self.num_neuron, self.learn_length));
    p_collector=np.zeros((1, self.learn_length));
    x=np.zeros((self.num_neuron, 1));
    
    for n in xrange(self.washout_length+self.learn_length):
      u=pattern[n];
      x_old=x;
      x=np.tanh(self.W_star.dot(x)+self.W_in*u+self.W_bias);
      if n>self.washout_length-1:
        x_collector[:, n-self.washout_length]=x[:,0];
        x_old_collector[:, n-self.washout_length]=x_old[:,0];
        p_collector[0, n-self.washout_length]=u;
    
    x_collector_centered=x_collector-numpy.matlib.repmat(np.mean(x_collector, 1), self.learn_length, 1).T;
    # document centered collectors and collectors
    self.x_collectors_centered.append(x_collector_centered);
    self.x_collectors.append(x_collector);
    
    # document eigen vectors and eigen values 
    R=x_collector.dot(x_collector.T)/self.learn_length;
    Ux, Sx, _=numpy.linalg.svd(R);
    self.sr_collectors.append(Sx);
    self.ur_collectors.append(Ux);
    self.pattern_rs.append(R);
    
    self.compute_projector(R);
    
    if not self.startXs.size:
      self.startXs=x[:,0];
      self.startXs=self.startXs[None].T;
    else:
      self.startXs=np.hstack((self.startXs, x));
    
    self.train_xpl.append(x_collector[:, 0:self.signal_plot_length]);
    self.train_ppl.append(p_collector[0, 0:self.signal_plot_length]);
    self.pattern_collectors.append(p_collector);
    
    # document training data
    if not self.all_train_args.size:
      self.all_train_args=x_collector;
    else:
      self.all_train_args=np.hstack((self.all_train_args, x_collector));
      
    if not self.all_train_old_args.size:
      self.all_train_old_args=x_old_collector;
    else:
      self.all_train_old_args=np.hstack((self.all_train_old_args, x_old_collector));
      
    if not self.all_train_outs.size:
      self.all_train_outs=p_collector;
    else:
      self.all_train_outs=np.hstack((self.all_train_outs, p_collector));
      
    if not self.all_train_targs.size:
      self.all_train_targs=self.W_in.dot(p_collector);
    else:
      self.all_train_targs=np.hstack((self.all_train_targs, self.W_in.dot(p_collector)));
      
    self.num_pattern+=1;
  
  def train(self,
            patterns):
    """
    Training procedure for conceptor network
    
    @param patterns: pattern list, the patterns are in column wise
    """
    if patterns.ndim==1:
      self.train_pattern(patterns);
    else:
      for i in xrange(patterns.shape[1]):
        self.train_pattern(patterns[:, i]);
    
    self.compute_weights(self.tychonov_alpha_readout);
    
  def compute_weights(self,
                      tychonov_alpha_readout=0.01):
    """
    Compute readout weights, target weights, and reservoir weights
    """
    self.compute_readout(self.tychonov_alpha_readout);
    self.compute_W(self.tychonov_alpha_readout);
  
  def compute_readout(self,
                      tychonov_alpha_readout=0.01):
    """
    Compute readout weight
    
    @param tychnonv_alpha_readout: Tychonov regularization parameter
    """
    
    self.W_out=np.linalg.inv(self.all_train_args.dot(self.all_train_args.T)+tychonov_alpha_readout*np.eye(self.num_neuron)).dot(self.all_train_args).dot(self.all_train_outs.T).T;
    
  def compute_W(self,
                tychonov_alpha_readout=0.01):
    """
    Compute reserior weights and target weights
    
    @param tychonov_alpha_readout: Tychonov regularization parameter
    """
    self.W_targets=np.arctanh(self.all_train_args)-numpy.matlib.repmat(self.W_bias, 1, self.num_pattern*self.learn_length);
    self.W=np.linalg.inv(self.all_train_old_args.dot(self.all_train_old_args.T)+tychonov_alpha_readout*np.eye(self.num_neuron)).dot(self.all_train_old_args).dot(self.W_targets.T).T;
    
  def messy_recall(self,
                   x,
                   test_length=200):
    """
    Run loaded reservior to observe a messy output.
    
    @param x: patterns restored in self.startXs
    @param test_length: length of recall
    """
    
    messy_out_pl=np.zeros((1, test_length));
    x=x[None].T;
    
    for n in xrange(test_length):
      x=np.tanh(self.W.dot(x)+self.W_bias);
      y=self.W_out.dot(x);
      messy_out_pl[:,n]=y[0,0];
      
    return messy_out_pl;
    
  def compute_projector(self,
                        R,
                        alpha=10):  
    """
    Compute projector (conceptor)
    
    @param R: state correlation matrix
    @param alpha: a designed parameter: aperture 
    """
    
    U,S,_=np.linalg.svd(R);
    S_new=(np.diag(S).dot(np.linalg.inv(np.diag(S)+alpha**(-2)*np.eye(self.num_neuron))))
        
    C=U.dot(S_new).dot(U.T);
    
    self.Cs[0].append(C);
    self.Cs[1].append(U);
    self.Cs[2].append(np.diag(S_new));
    self.Cs[3].append(S);
    
    
class Autoconceptor:
  """
  A implementation of Autoconceptor
  
  This class is designed by following:
  Section 3.13 Autoconceptors in
  Controlling Recurrent Neural Networks by Conceptors
  """
  
  def __init__(self,
               num_neuron):
    """
    comments
    """
    
    
    pass
    
    
    
class RandomFeatureConceptor:
  """
  An implementation of Random Feature Conceptor
  
  This class is designed by following:
  3.14 Toward Biologically Plausible Neural Circuits: Random Feature Conceptors in
  Controlling Recurrent Neural Networks by Conceptors
  """
  
  def __init__(self,
               num_neuron):
    """
    Write things
    """
  
    pass

    
    
    
    
    
    
    
    