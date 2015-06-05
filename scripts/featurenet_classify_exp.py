"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: FeatureNet Classification Test
"""

import sys;
sys.path.append("..");

import time;

import numpy as np;
from sacred import Experiment;

import conceptors.util as util;
from conceptors.feature_net import FeatureNet;
from conceptors.dataset import load_arlab_feature;

exp=Experiment("FeatureNet Classification Task");

@exp.config
def classify_exp_config():
  filename_train="";
  filename_test="";
  save_path="";
  ap_N=10;
  num_inter_samples=900;

@exp.automain
def classify_experiment(filename_train,
                        filename_test,
                        save_path,
                        ap_N,
                        num_inter_samples):
  """
  Supervised Classification Task
  
  @param filename_train: train data file
  @param filename_test: test data file
  @param save_path: result save path
  """
  
  # Load data
  train_data, test_data=load_arlab_feature(filename_train, filename_test);
  
  # Global paramete settings
  num_classes=int(np.max(train_data[:,0])+1);
  num_train=train_data.shape[0];
  num_test=test_data.shape[0];
  
  train_input, train_label, test_input, test_label=util.parse_arlab_feature(train_data, test_data);
  
  train_input, test_input=util.normalize_arlab_feature(train_input, test_input);
  train_input=train_input.T;
  test_input=test_input.T;
  _, tr_start_idx=np.unique(train_label, return_index=True);
  
  print "[MESSAGE] Data is prepared.";
  
  start_time=time.clock();
  save_file=open(save_path, "a+");
  feature_size=train_input.shape[0];
  
  network=FeatureNet(feature_size=feature_size);
  
  def calculate_block(block, num_classes):
      if block==num_classes-1:
        start_idx=tr_start_idx[block];
        end_idx=train_input.shape[1];
      else:
        start_idx=tr_start_idx[block];
        end_idx=tr_start_idx[block+1];
      return start_idx, end_idx;
    
  C_poss=[]; R_poss=[];
  for block in xrange(num_classes):
    start_idx, end_idx=calculate_block(block, num_classes);
      
    C_pos_class, R=network.compute_conceptor(train_input[:, start_idx:end_idx],
                                             ap_N,
                                             (num_train-(end_idx-start_idx)));
    
    C_poss.append(C_pos_class);
    R_poss.append(R);
    
  print "[MESSAGE] Conceptors Computed"
  
  best_aps_poss=np.zeros(num_classes);
  for i in xrange(num_classes):
    best_aps_poss[i]=network.compute_pos_aperture(C_poss[i],
                                                  ap_N,
                                                  num_inter_samples);
    
  best_ap_pos=np.mean(best_aps_poss);
  
  print "[MESSAGE] Best Positive Aperture: %.2f" % (best_ap_pos);
  
  C_pos_best=[];
  for block in xrange(num_classes):
    start_idx, end_idx=calculate_block(block, num_classes);
      
    c_pos_best=network.compute_pos_conceptor(R_poss[block],
                                             best_ap_pos,
                                             end_idx-start_idx);
    C_pos_best.append(c_pos_best);
      
      
  print "[MESSAGE] Best conceptors computed"
  
  
  xTx=test_input.T.dot(test_input).diagonal();
        
  pos_ev=np.zeros((num_classes, num_test));
    
  for i in xrange(num_classes):
    for j in xrange(num_test):
        pos_ev[i,j]=test_input[:,j].dot(C_pos_best[i]).dot(test_input[:,j][None].T)/xTx[j];
        
  print "[MESSAGE] %i class evidence is calculated" % (i+1);
  
  pos_out_label=np.argmax(pos_ev, axis=0);
  
  pos_accuracy=float(np.sum(pos_out_label==test_label))/float(num_test);
  
  print "[MESSAGE] Pos Accuracy %.2f %%" % (pos_accuracy*100);
  
  end_time=time.clock();
  print "[MESSAGE] Total for %.2fm" % ((end_time-start_time)/60);

  info=np.column_stack((best_ap_pos, best_ap_pos, pos_accuracy, ((end_time-start_time)/60)));
  np.savetxt(save_file, info, delimiter=',',newline='\n');
  save_file.close();