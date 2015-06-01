'''
Created on Feb 25, 2015

@author: Yuhuang Hu
@note: a concept network implementation
'''

import numpy as np;
import numpy.matlib;
import scipy.interpolate;

import conceptors.util;
import conceptors.logic as logic;

class ConceptorNetwork:
  """
  A implementaion of conceptor network
  
  This class is designed by following:
  Section 3.2 Driving a Reservorir with Different Patterns in
  Controlling Recurrent Neural Networks by Conceptors
  """
  
  def __init__(self,
               num_in,
               num_neuron,
               sr=1.5,
               in_scale=1.5,
               bias_scale=0.2,
               washout_length=500,
               learn_length=1000,
               signal_plot_length=20,
               tychonov_alpha_readout=0.01,
               tychonov_alpha_readout_w=0.0001):
    
    """
    Initialize conceptor network
    
    @param num_in: number of input neurons
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
    self.num_in=num_in;
    self.num_neuron=num_neuron;
    self.num_pattern=0;
    self.sr=sr;
    self.in_scale=in_scale;
    self.bias_scale=bias_scale;
    self.learn_length=learn_length;
    self.washout_length=washout_length;
    self.signal_plot_length=signal_plot_length;
    self.tychonov_alpha_readout=tychonov_alpha_readout;
    self.tychonov_alpha_readout_w=tychonov_alpha_readout_w;
    
    # initialize weights and parameters
    W_star, W_in, W_bias=conceptors.util.init_weights(num_in,
                                                      num_neuron,
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
    self.I=np.eye(self.num_neuron+self.num_in);
    
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
    
  def drive_class(self,
                  patterns):
    """
    @param patterns: Class of patterns
    """
    
    x=np.random.rand(self.num_neuron, 1);
     
    hid_states=np.array([]);
    for i in xrange(patterns.shape[1]):
      x=np.tanh(self.W_star.dot(x)+self.W_in.dot(patterns[:,i][None].T)+self.W_bias);
       
      x_seq=np.vstack((x, patterns[:,i][None].T));
      if not hid_states.size:
        hid_states=x_seq;
      else:
        hid_states=np.hstack((hid_states, x_seq));
    
    return hid_states;
     
#     x=np.random.rand(self.num_neuron,
#                      patterns.shape[1]);
#     x=np.tanh(self.W_star.dot(x)+self.W_in.dot(patterns)+self.W_bias);
#     
#     return np.vstack((x, patterns));
  
  def compute_conceptor(self,
                        train_states,
                        ap_N,
                        R_all,
                        norm_size):
    """
    @param train_states: a class of training state
    @param R_all: Correlation matrix for all training states
    @param norm_size: norm size for R_orther_norm;
    """
    
    R=train_states.dot(train_states.T);
    R_norm=R/float(train_states.shape[1]);
    R_other=R_all-R;
    R_other_norm=R_other/(float(norm_size));
    
    C_pos_class=[];
    C_neg_class=[];
    
    for ap_i in xrange(ap_N):
      C_pos_class.append(R_norm.dot(np.linalg.inv(R_norm+(2**float(ap_i))**(-2)*self.I)));
      C_other=R_other_norm.dot(np.linalg.inv(R_other_norm+(2**float(ap_i))**(-2)*self.I));
      C_neg_class.append(self.I-C_other);
      
    return C_pos_class, C_neg_class, R, R_other;
  
  def compute_aperture(self,
                       C_pos_class,
                       C_neg_class,
                       ap_N,
                       num_inter_samples):
    """
    @param C_pos_class: a class of positive conceptors
    @param C_neg_class: a class of negative conceptors 
    """
    
    norm_pos=np.zeros(ap_N);
    norm_neg=np.zeros(ap_N);
    for ap_i in xrange(ap_N):
      norm_pos[ap_i]=np.linalg.norm(C_pos_class[ap_i], 'fro')**2;
      norm_neg[ap_i]=np.linalg.norm(self.I-C_neg_class[ap_i], 'fro')**2;
      
    f_pos=scipy.interpolate.interp1d(np.arange(ap_N), norm_pos, kind="cubic");
    x_new=np.linspace(0, ap_N-1, num_inter_samples+1);
    f_neg=scipy.interpolate.interp1d(np.arange(ap_N), norm_neg, kind="cubic");
    norm_pos_inter=f_pos(x_new);
    norm_neg_inter=f_neg(x_new);
    
    norm_pos_inter_grad=(norm_pos_inter[1:]-norm_pos_inter[0:-1])/0.01;
    #norm_pos_inter_grad=np.hstack((norm_pos_inter_grad, norm_pos_inter_grad[-1]));
    
    norm_neg_inter_grad=(norm_neg_inter[1:]-norm_neg_inter[0:-1])/0.01;
    #norm_neg_inter_grad=np.hstack((norm_neg_inter_grad, norm_neg_inter_grad[-1]));
    
    max_ind_pos=np.argmax(np.abs(norm_pos_inter_grad));
    max_ind_neg=np.argmax(np.abs(norm_neg_inter_grad));
    
    best_aps_pos=2**x_new[max_ind_pos];
    best_aps_neg=2**x_new[max_ind_neg];
    
    return best_aps_pos, best_aps_neg;
  
  def compute_best_conceptor(self,
                             R,
                             R_other,
                             best_ap_pos,
                             best_ap_neg,
                             norm_size,
                             norm_others_size):
    R_norm=R/float(norm_size);
    R_other_norm=R_other/float(norm_others_size);
    
    c_pos_best=R_norm.dot(np.linalg.inv(R_norm + best_ap_pos ** (-2) * self.I));
    C_other=R_other_norm.dot(np.linalg.inv(R_other_norm + best_ap_neg ** (-2) * self.I));
    c_neg_best=self.I-C_other;
      
    return c_pos_best, c_neg_best;
  
  def compute_pos_conceptor(self,
                            R,
                            ap_pos,
                            norm_size):
    R_norm=R/float(norm_size);
    c_pos=R_norm.dot(np.linalg.inv(R_norm + ap_pos ** (-2) * self.I));
    
    return c_pos;
      
  def train_pattern(self,
                    pattern):
    """
    This function train one single input pattern.
    
    @param pattern: input pattern
    """
    
    x_collector=np.zeros((self.num_neuron, self.learn_length));
    x_old_collector=np.zeros((self.num_neuron, self.learn_length));
    p_collector=np.zeros((self.num_in, self.learn_length));
    x=np.zeros((self.num_neuron, 1));
    
    for n in xrange(self.washout_length+self.learn_length):
      u=pattern[:,n][None].T;
      x_old=x;
      x=np.tanh(self.W_star.dot(x)+self.W_in.dot(u)+self.W_bias);
      if n>self.washout_length-1:
        x_collector[:, n-self.washout_length]=x[:,0];
        x_old_collector[:, n-self.washout_length]=x_old[:,0];
        p_collector[:, n-self.washout_length]=u[:,0];
    
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
    self.train_ppl.append(p_collector[:, 0:self.signal_plot_length]);
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
    
    @param patterns: pattern list, each pattern is a element in the list
    """
    for i in xrange(len(patterns)):
      self.train_pattern(patterns[i]);
    
    self.compute_weights(self.tychonov_alpha_readout,
                         self.tychonov_alpha_readout_w);
    
  def compute_weights(self,
                      tychonov_alpha_readout=0.01,
                      tychonov_alpha_readout_w=0.0001):
    """
    Compute readout weights, target weights, and reservoir weights
    """
    self.compute_readout(self.tychonov_alpha_readout);
    self.compute_W(self.tychonov_alpha_readout_w);
  
  def compute_readout(self,
                      tychonov_alpha_readout=0.01):
    """
    Compute readout weight
    
    @param tychnonv_alpha_readout: Tychonov regularization parameter
    """
    
    self.W_out=np.linalg.inv(self.all_train_args.dot(self.all_train_args.T)+tychonov_alpha_readout*np.eye(self.num_neuron)).dot(self.all_train_args).dot(self.all_train_outs.T).T;
    
  def compute_W(self,
                tychonov_alpha_readout_w=0.0001):
    """
    Compute reserior weights and target weights
    
    @param tychonov_alpha_readout: Tychonov regularization parameter
    """
    self.W_targets=np.arctanh(self.all_train_args)-numpy.matlib.repmat(self.W_bias, 1, self.num_pattern*self.learn_length);
    self.W=np.linalg.inv(self.all_train_old_args.dot(self.all_train_old_args.T)+tychonov_alpha_readout_w*np.eye(self.num_neuron)).dot(self.all_train_old_args).dot(self.W_targets.T).T;
    
  def messy_recall(self,
                   x,
                   test_length=200):
    """
    Run loaded reservior to observe a messy output.
    
    @param x: patterns restored in self.startXs
    @param test_length: length of recall
    """
    
    messy_out_pl=np.zeros((self.num_in, test_length));
    x=x[None].T;
    
    for n in xrange(test_length):
      x=np.tanh(self.W.dot(x)+self.W_bias);
      y=self.W_out.dot(x);
      messy_out_pl[:,n]=y[:,0];
      
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
               num_in,
               num_neuron,
               sr=1.5,
               in_scale=1.5,
               bias_scale=0.5,
               washout_length=100,
               learn_length=500,
               measure_template_length=20,
               signal_plot_length=10,
               alpha=100,
               tychonov_alpha_d=0.001,
               tychonov_alpha_w_out=0.0001):
    """
    Initialize Autoconceptor Network
    
    @param num_in: number of input neurons
    @param num_neuron: number of neurons in reservoir
    @param sr: spectral radius
    @param in_scale: scaling of pattern feeding weights
    @param bias_scale: size of bias 
    @param washout_length: length of wash-out iteration
    @param learn_length: length of learning iteration
    @param signal_plot_length: length of plot length
    @param alpha: aperture
    @param tychnonv_alpha_readout: Tychonov regularization parameter
    """
    
    self.num_in=num_in;
    self.num_neuron=num_neuron;
    self.sr=sr;
    self.in_scale=in_scale;
    self.bias_scale=bias_scale;
    self.learn_length=learn_length;
    self.washout_length=washout_length;
    self.measure_template_length=measure_template_length;
    self.signal_plot_length=signal_plot_length;
    self.alpha=alpha;
    self.tychonov_alpha_d=tychonov_alpha_d;
    self.tychonov_alpha_w_out=tychonov_alpha_w_out;
    
    # initialize weights
    W, W_in, bias=conceptors.util.init_weights(num_in,
                                               num_neuron,
                                               sr,
                                               in_scale,
                                               bias_scale);
                                                      
    self.W=W;
    self.W_in=W_in;
    self.W_out=np.asarray([]);
    self.bias=bias;
    
    self.p_templates=np.asarray([]);
    self.all_train_args=np.asarray([]);
    self.all_train_old_args=np.asarray([]);
    self.all_train_dt_args=np.asarray([]);
    self.all_train_yt_args=np.asarray([]);
    
    # input simulation matrix
    self.D=np.zeros((self.num_neuron, self.num_neuron));
    
    # conceptor matrix
    self.C=np.zeros((self.num_neuron, self.num_neuron));
    
  def load_pattern(self,
                   pattern,
                   incremental=True):
    """
    Load single pattern
    
    @param pattern: input pattern in (N x time) size, each column is a sample
    @param incremental: if enable incremental learning
    """
    
    p_template=np.zeros((self.learn_length, self.num_in));
    x_cue=np.zeros((self.num_neuron, self.learn_length));
    x_old_cue=np.zeros((self.num_neuron, self.learn_length));
    #p_collector=np.zeros(())
    x=np.zeros((self.num_neuron,1));
    
    for n in xrange(self.washout_length):
      u=pattern[:,n][None].T;
      x=np.tanh(self.W.dot(x)+self.W_in.dot(u)+self.bias);
    
    for n in xrange(self.learn_length):
      u=pattern[:,n+self.washout_length][None].T;
      x_old_cue[:,n]=x[:,0];
      x=np.tanh(self.W.dot(x)+self.W_in.dot(u)+self.bias);
      x_cue[:,n]=x[:,0];
      p_template[n,:]=u[:,0].T;
      
    if not self.p_templates.size:
      self.p_templates=p_template[-1-self.measure_template_length+1:,:];
    else:
      self.p_templates=np.hstack((self.p_templates, p_template[-1-self.measure_template_length+1:,:]));
      
    if not self.all_train_args.size:
      self.all_train_args=x_cue;
    else:
      self.all_train_args=np.hstack((self.all_train_args, x_cue));
      
    if not self.all_train_old_args.size:
      self.all_train_old_args=x_old_cue;
    else:
      self.all_train_old_args=np.hstack((self.all_train_old_args, x_old_cue));
      
    if not self.all_train_dt_args.size:
      self.all_train_dt_args=self.W_in.dot(p_template.T);
    else:
      self.all_train_dt_args=np.hstack((self.all_train_dt_args, self.W_in.dot(p_template.T)));
      
    if not self.all_train_yt_args.size:
      self.all_train_yt_args=p_template.T;
    else:
      self.all_train_yt_args=np.hstack((self.all_train_yt_args, p_template.T));
      
    if incremental==True:
      self.update_input_simulation_matrix(p_template, x_old_cue);
      self.update_conceptor(x_old_cue);
      
  def load(self,
           patterns,
           load_mode="incremental"):
    """
    This function loads patterns
    
    @param patterns: list of patterns that are loaded.
    @param load_mode: pattern loading mode ("incermental"/"complete")
    """
    
    if load_mode=="incremental":
      for i in xrange(len(patterns)):
        self.load_pattern(patterns[i]);
    elif load_mode=="complete":
      for i in xrange(len(patterns)):
        self.load_pattern(patterns[i], False);
      self.learn_input_simulation_matrix(self.tychonov_alpha_d);
    
    self.learn_output_weights(self.tychonov_alpha_w_out);
      
      
  def learn_input_simulation_matrix(self,
                                    tychonov_alpha_d=0.001):
    """
    Direct learn input simulation matrix D
    
    @param tychonov_alpha_d: Tychonov regularization parameter 
    """
    
    self.D=np.linalg.inv(self.all_train_old_args.dot(self.all_train_old_args.T)+tychonov_alpha_d*np.eye(self.num_neuron)).dot(self.all_train_old_args).dot(self.all_train_dt_args.T).T;
    
  def learn_output_weights(self,
                           tychonov_alpha_w_out=0.01):
    """
    Compute outpout weights
    
    @param tychonov_alpha_w_out: Tychonov regularization parameter
    """
    
    self.W_out=np.linalg.inv(self.all_train_args.dot(self.all_train_args.T)+self.tychonov_alpha_w_out*np.eye(self.num_neuron)).dot(self.all_train_args).dot(self.all_train_yt_args.T).T;
    
  def update_input_simulation_matrix(self,
                                     p_template,
                                     x_old_cue):
    """
    Incremental learning of input simulation matrix
    
    @param p_template: 
    @param x_old_collector: 
    """
    
    D_targs=self.W_in.dot(p_template.T)-self.D.dot(x_old_cue);
    F=logic.NOT(self.C);
    D_args=F.dot(x_old_cue);
    D_inc=(np.linalg.pinv(D_args.dot(D_args.T)/self.learn_length+self.alpha**-2*np.eye(self.num_neuron)).dot(D_args).dot(D_targs.T)/self.learn_length).T;
    
    self.D+=D_inc;
    
  def update_conceptor(self,
                       x_old_cue):
    """
    Incremental learning of conceptor
    
    @param x_old_cue: 
    """
    
    R=x_old_cue.dot(x_old_cue.T)/(self.learn_length+1);
    C_native=R.dot(np.linalg.inv(R+np.eye(self.num_neuron)));
    C_ap=logic.PHI(C_native, self.alpha);
    
    self.C=logic.OR(self.C, C_ap);
    
  def adapt_conceptor(self,
                      C,
                      x,
                      lambda_adapt):
    """
    Adapt conceptor matrix
    
    @param C: current concepotr matrix
    @param x: current activiations
    @param lambda_adapt: adapt parameter
    
    @return next state of conceptor matrix
    """
    
    return lambda_adapt*((x-C.dot(x)).dot(x.T)-(self.alpha**-2)*C);
    
  def cue_conceptor(self,
                    pattern,
                    init_washout=100,
                    cue_length=30,
                    lambda_adapt_cue=0.01):
    """
    Cue conceptor in following three stages:
    1. Initial washout
    2. Cueing
    
    @param pattern: input pattern
    @param init_washout: initial washout length
    @param cue_length: cueing length;
    """
    
    x=np.zeros((self.num_neuron,1));
    for n in xrange(init_washout):
      u=pattern[:,n][None].T;
      x=np.tanh(self.W.dot(x)+self.W_in.dot(u)+self.bias);
      
    C=np.zeros((self.num_neuron, self.num_neuron));
    
    for n in xrange(cue_length):
      u=pattern[:,n+init_washout][None].T;
      x=np.tanh(self.W.dot(x)+self.W_in.dot(u)+self.bias);
      C+=self.adapt_conceptor(C, x, lambda_adapt_cue);
    
    return C, x;
  
  def recall_conceptor(self,
                       C,
                       x,
                       adaptation_length=10000,
                       lambda_adapt_recall=0.01):
    """
    Autonomous recall pattern based on conceptor
    
    @param C: current cue conceptor matrix
    @param x: current activation state
    @param lambda_adapt_recall: recall apdaptation parameter
    
    @return Conceptor matrix from recall
    """
    
    for n in xrange(adaptation_length):
      x=C.dot(np.tanh(self.W.dot(x)+self.D.dot(x)+self.bias));
      C+=self.adapt_conceptor(C, x, lambda_adapt_recall);
      
    return C, x;
  
  def offline_recall(self,
                     C,
                     x):
    """
    Activation offline recall using conceptor C and an activation state x
    
    @param C: conceptor matrix
    @param x: activation state
    """
    
    r=np.tanh(self.W.dot(x)+self.D.dot(x)+self.bias);
    
    return C.dot(r), r;
  
  def offline_pattern_recall(self,
                             C,
                             x):
    """
    Pattern offline recall using conceptor C and an activation state x
    
    @param C: conceptor matrix
    @param x: activation state 
    """
    
    x, r=self.offline_recall(C, x);
    
    return self.W_out.dot(r), x;
    
class RandomFeatureConceptor:
  """
  An implementation of Random Feature Conceptor
  
  This class is designed by following:
  3.14 Toward Biologically Plausible Neural Circuits: Random Feature Conceptors in
  Controlling Recurrent Neural Networks by Conceptors
  """
  
  def __init__(self,
               num_in,
               num_neuron,
               num_feature,
               sr=1.4,
               in_scale=1.2,
               bias_scale=0.2,
               washout_length=200,
               learn_length=400,
               c_adapt_length=2000,
               signal_plot_length=20,
               alpha=8,
               tychonov_alpha_readout=1,
               tychonov_alpha_g=0.01):
    """
    Initialize a Random Feature Conceptor network
    
    @param num_in: number of input neurons
    @param num_neuron: number of neurons in reservoir
    @param num_feature: number of neurons in feature space
    @param sr: spectral radius
    @param in_scale: scaling of pattern feeding weights
    @param bias_scale: size of bias
    """
  
    self.num_in=num_in;
    self.num_neuron=num_neuron;
    self.num_feature=num_feature;
    self.sr=sr;
    self.in_scale=in_scale;
    self.bias_scale=bias_scale;
    self.washout_length=washout_length;
    self.learn_length=learn_length;
    self.c_adapt_length=c_adapt_length;
    self.signal_plot_length=signal_plot_length;
    self.alpha=alpha;
    self.tychonov_alpha_readout=tychonov_alpha_readout;
    self.tychonov_alpha_g=tychonov_alpha_g;
    
    # initialize weights
    
    _, W_in, bias=conceptors.util.init_weights(num_in,
                                               num_neuron,
                                               sr,
                                               in_scale,
                                               bias_scale);
    self.W_in=W_in;
    self.bias=bias;
    self.W_out=np.asarray([]);
    
    F_raw=np.random.rand(num_feature, num_neuron);
    G_star_raw=np.random.rand(num_neuron, num_feature);
    GF=G_star_raw.dot(F_raw);
    SR=np.max(np.abs(np.linalg.eigvals(GF)));
    F_raw/=np.sqrt(SR);
    G_star_raw/=np.sqrt(SR);
    
    self.F=F_raw*np.sqrt(sr);
    self.G_star=G_star_raw*np.sqrt(sr);
    self.G=np.asarray([]);
    
    self.all_train_r=np.asarray([]);
    self.all_train_cz_old=np.asarray([]);
    self.all_train_p=np.asarray([]);
    self.all_train_t=np.asarray([]);
    
    # collectors
    
    self.c_collectors=[];
    self.Cs=[];
    self.train_rpl=[];
    self.train_ppl=[];
    
  def load(self,
           pattern):
    """
    Load a pattern
    
    @param pattern: a pattern to be loaded, each column is a sample
    """

    r_collector=np.zeros((self.num_neuron, self.learn_length));
    cz_old_collector=np.zeros((self.num_feature, self.learn_length));
    p_collector=np.zeros((self.num_in, self.learn_length));
    c_collector=np.zeros((self.num_feature, self.c_adapt_length));
    t_collector=np.zeros((self.num_neuron, self.learn_length));
    
    z=np.zeros((self.num_feature, 1));
    c=np.ones((self.num_feature, 1));
    cz=np.zeros((self.num_feature, 1));
    
    for n in xrange(self.washout_length+self.c_adapt_length+self.learn_length):
      u=pattern[:,n][None].T;
      cz_old=cz;
      t=self.G_star.dot(cz)+self.W_in.dot(u);
      r=np.tanh(t+self.bias);
      z=self.F.dot(r);
      cz=c*z;
      
      if (n<=self.c_adapt_length+self.washout_length-1 and n>self.washout_length-1):
        c=c+self.update_conception(c, cz);
        c_collector[:,n-self.washout_length]=c;
        
      if n==self.c_adapt_length+self.washout_length-1:
        self.Cs.append(c);
      
      if n>self.c_adapt_length+self.washout_length-1:
        r_collector[:, n-self.washout_length-self.c_adapt_length]=r;
        cz_old_collector[:, n-self.washout_length-self.c_adapt_length]=cz_old;
        p_collector[:, n-self.washout_length-self.c_adapt_length]=u;
        t_collector[:, n-self.washout_length-self.c_adapt_length]=t;
        
    self.c_collectors.append(c_collector);
    self.train_rpl.append(r_collector[0:5, 0:self.signal_plot_length]);
    self.train_ppl.append(p_collector[:, 0:self.signal_plot_length]);
    
    if not self.all_train_r.size:
      self.all_train_r=r_collector;
    else:
      self.all_train_r=np.hstack((self.all_train_r, r_collector));
      
    if not self.all_train_cz_old.size:
      self.all_train_cz_old=cz_old_collector;
    else:
      self.all_train_cz_old=np.hstack((self.all_train_cz_old, cz_old_collector));
      
    if not self.all_train_p.size:
      self.all_train_p=p_collector;
    else:
      self.all_train_p=np.hstack((self.all_train_p, p_collector));
      
    if not self.all_train_t.size:
      self.all_train_t=t_collector;
    else:
      self.all_train_t=np.hstack((self.all_train_t, t_collector));
    
  def update_conception(self,
                        c,
                        cz,
                        lambda_conception=0.5):
    """
    Update conception vector
    """
    
    return lambda_conception*((cz-c*cz)*cz-self.alpha**-2*c);
    
  def compute_readout_weights(self,
                              tychonov_alpha_readout=1):
    """
    Compute readout weights
    """
    
    self.W_out=np.linalg.inv(self.all_train_r.dot(self.all_train_r.T)+tychonov_alpha_readout*np.eye(self.num_neuron)).dot(self.all_train_r).dot(self.all_train_p.T).T;
    
  def compute_backprojection_weights(self,
                                     tychonov_alpha_g=0.01):
    """
    Compute backprojection weights
    """
    
    self.G=np.linalg.inv(self.all_train_cz_old.dot(self.all_train_cz_old.T)+tychonov_alpha_g*np.eye(self.num_feature)).dot(self.all_train_cz_old).dot(self.all_train_t.T).T;
  