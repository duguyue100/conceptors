"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: setup experiment of supervised classification task.
"""

import sys;
sys.path.append("..");

import time;

import numpy as np;
from sacred import Experiment;

import conceptors.util as util;
import conceptors.net as net;
from conceptors.dataset import load_arlab_feature;

exp=Experiment("Classification Task");

@exp.config
def classify_exp_config():
  filename_train="";
  filename_test="";
  save_path="";
  range_start=5;
  range_end=500;
  range_step=5;
  ap_N=10;
  num_inter_samples=900;

@exp.automain
def classify_experiment(filename_train,
                        filename_test,
                        save_path,
                        range_start,
                        range_end,
                        range_step,
                        ap_N,
                        num_inter_samples):
  """
  Supervised Classification Task
  
  @param filename_train: train data file
  @param filename_test: test data file
  @param save_path: result save path
  @param range_start: number, start number of neuron range
  @param range_end: number, end number of neuron range
  @param range_step: number, neuron step
  """
  
  # Load data
  train_data, test_data=load_arlab_feature(filename_train, filename_test);
  
  # Global paramete settings
  num_classes=int(np.max(train_data[:,0])+1);
  num_train=train_data.shape[0];
  num_test=test_data.shape[0];
  
  neuron_range=np.arange(range_start, range_end+range_step, range_step);
  
  train_input, train_label, test_input, test_label=util.parse_arlab_feature(train_data, test_data);
  
  train_input, test_input=util.normalize_arlab_feature(train_input, test_input);
  train_input=train_input.T;
  test_input=test_input.T;
  _, tr_start_idx=np.unique(train_label, return_index=True);
  _, te_start_idx=np.unique(test_label, return_index=True);
  
  print "[MESSAGE] Data is prepared.";
  
  for trail in xrange(len(neuron_range)):
    print "[MESSAGE] Trail %d" % (trail+1);
    start_time=time.clock();
    
    ## parameter settings
    save_file=open(save_path, "a+");
    num_in=train_input.shape[0];
    num_neuron=neuron_range[trail];
    
    ## create network
    network=net.ConceptorNetwork(num_in=num_in,
                                 num_neuron=num_neuron,
                                 sr=1.5,
                                 in_scale=1.5,
                                 bias_scale=0.2,
                                 washout_length=0,
                                 learn_length=1,
                                 signal_plot_length=0,
                                 tychonov_alpha_readout=0.01,
                                 tychonov_alpha_readout_w=0.0001);
    
    def calculate_block(block, num_classes):
      if block==num_classes-1:
        start_idx=tr_start_idx[block];
        end_idx=train_input.shape[1];
      else:
        start_idx=tr_start_idx[block];
        end_idx=tr_start_idx[block+1];
      return start_idx, end_idx;
    
    all_train_states=np.array([]);
    for block in xrange(num_classes):
      start_idx, end_idx=calculate_block(block, num_classes);
      
      temp_train_states=network.drive_class(train_input[:, start_idx:end_idx]);
      if not all_train_states.size:
        all_train_states=temp_train_states;
      else:
        all_train_states=np.hstack((all_train_states,temp_train_states));
    print "[MESSAGE] Train data driven"
    
    R_all=all_train_states.dot(all_train_states.T);
    C_poss=[]; C_negs=[]; R_poss=[]; R_others=[];
    for block in xrange(num_classes):
      start_idx, end_idx=calculate_block(block, num_classes);
      
      C_pos_class, C_neg_class, R, R_other=network.compute_conceptor(all_train_states[:, start_idx:end_idx],
                                                                     ap_N,
                                                                     R_all,
                                                                     (num_train-(end_idx-start_idx)));
    
      C_poss.append(C_pos_class);
      C_negs.append(C_neg_class);
      R_poss.append(R);
      R_others.append(R_other);
    print "[MESSAGE] Conceptors Computed"
    
    best_aps_poss=np.zeros(num_classes);
    best_aps_negs=np.zeros(num_classes);
    for i in xrange(num_classes):
      best_aps_poss[i], best_aps_negs[i]=network.compute_aperture(C_poss[i],
                                                                  C_negs[i],
                                                                  ap_N,
                                                                  num_inter_samples);
    
    best_ap_pos=np.mean(best_aps_poss);
    best_ap_neg=np.mean(best_aps_negs);                       
    
    print "[MESSAGE] Best Positive Aperture: %.2f, Best Negative Aperture: %.2f" % (best_ap_pos, best_ap_neg);
    
    C_pos_best=[]; C_neg_best=[];
    for block in xrange(num_classes):
      start_idx, end_idx=calculate_block(block, num_classes);
      
      c_pos_best, c_neg_best=network.compute_best_conceptor(R_poss[block],
                                                            R_others[block],
                                                            best_ap_pos,
                                                            best_ap_neg,
                                                            end_idx-start_idx,
                                                            num_train-(end_idx-start_idx));
      C_pos_best.append(c_pos_best);
      C_neg_best.append(c_neg_best);
      
    print "[MESSAGE] Best conceptors computed"
    
    x_test=network.drive_class(test_input);
    xTx=x_test.T.dot(x_test).diagonal();
        
    pos_ev=np.zeros((num_classes, num_test));
    neg_ev=np.zeros((num_classes, num_test));
    comb_ev=np.zeros((num_classes, num_test));
    
    for i in xrange(num_classes):
      for j in xrange(num_test):
        pos_ev[i,j]=x_test[:,j].dot(C_pos_best[i]).dot(x_test[:,j][None].T)/xTx[j];
        neg_ev[i,j]=x_test[:,j].dot(C_neg_best[i]).dot(x_test[:,j][None].T)/xTx[j];
        comb_ev[i,j]=pos_ev[i,j]+neg_ev[i,j];
        
      print "[MESSAGE] %i class evidence is calculated" % (i+1);
    
    output_label=np.argmax(comb_ev, axis=0);
    
    accuracy=float(np.sum(output_label==test_label))/float(num_test);
    
    print "[MESSAGE] Accuracy %.2f %%" % (accuracy*100);
    
    end_time=time.clock();
    print "[MESSAGE] Total for %.2fm" % ((end_time-start_time)/60);