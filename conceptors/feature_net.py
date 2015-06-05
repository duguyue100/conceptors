"""
@author: YuhuangHu
@contact: duguyue100@gmail.com

@note: First attempt of FeatureNet
"""

import numpy as np;
import scipy.interpolate;

class FeatureNet(object):
  """
  Implementation of FeatureNet
  """
  
  def __init__(self,
               feature_size):
    """
    init a feature net
    
    @param feature_size: size of input feature
    """
    
    self.feature_size=feature_size;
    
    self.I=np.eye(feature_size);
    
  def compute_conceptor(self,
                        feature_states,
                        ap_N,
                        norm_size):
    """
    @param feature_states: a class of training state
    @param R_all: Correlation matrix for all training states
    @param norm_size: norm size for R_orther_norm;
    """
    
    R=feature_states.dot(feature_states.T);
    R_norm=R/float(feature_states.shape[1]);
    
    C_pos_class=[];
    
    for ap_i in xrange(ap_N):
      C_pos_class.append(R_norm.dot(np.linalg.inv(R_norm+(2**float(ap_i))**(-2)*self.I)));
      
    return C_pos_class, R;
    
  def compute_pos_aperture(self,
                           C_pos_class,
                           ap_N,
                           num_inter_samples):
    """
    @param C_pos_class: a class of positive conceptors
    @param ap_N:
    @param num_inter_samples: number of interpolate points
    
    @return best positive learning rate
    """
    
    norm_pos=np.zeros(ap_N);
    for ap_i in xrange(ap_N):
      norm_pos[ap_i]=np.linalg.norm(C_pos_class[ap_i], 'fro')**2;
      
    f_pos=scipy.interpolate.interp1d(np.arange(ap_N), norm_pos, kind="cubic");
    x_new=np.linspace(0, ap_N-1, num_inter_samples+1);
    norm_pos_inter=f_pos(x_new);
    
    norm_pos_inter_grad=(norm_pos_inter[1:]-norm_pos_inter[0:-1])/0.01;
    max_ind_pos=np.argmax(np.abs(norm_pos_inter_grad));
    best_aps_pos=2**x_new[max_ind_pos];
    
    return best_aps_pos;
    
  def compute_pos_conceptor(self,
                            R,
                            ap_pos,
                            norm_size):
    """
    compute positive conceptor
    
    @param R: correlation matrix for the class
    @param ap_pos: learning rate for positive conceptor
    @param norm_size: normalization fator (may not be needed)
    
    @return: positive conceptors 
    """
    R_norm=R/float(norm_size);
    c_pos=R_norm.dot(np.linalg.inv(R_norm + ap_pos ** (-2) * self.I));
    
    return c_pos;