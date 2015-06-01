"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: setup experiment of incremental learning task.
"""

import sys;
sys.path.append("..");

import time;

import numpy as np;
from sacred import Experiment;

import conceptors.util as util;
import conceptors.net as net;
from conceptors.dataset import load_arlab_feature;

exp=Experiment("Incremental Learning Task");

@exp.config
def inc_learn_exp_config():
  filename_train="";
  filename_test="";
  save_path="";
  num_neuron=5;
  ap_pos=0;

@exp.automain
def inc_learn_exp(filename_train,
                  filename_test,
                  save_path,
                  num_neuron,
                  ap_pos):
  """
  Setup incremental learning task
  
  @param filename_train: train data file
  @param filename_test: test data file
  @param save_path: result save path
  @param num_neuron: numbe of hidden neurons
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
  _, te_start_idx=np.unique(test_label, return_index=True);
  
  print "[MESSAGE] Data is prepared.";
  
  # setup network
  
  save_file=open(save_path, "a+");
  num_in=train_input.shape[0];
  
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
                               
  def calculate_block(set_idx, block, num_classes, dataset="train"):
      if block==num_classes-1:
        start_idx=set_idx[block];
        if dataset=="train":
          end_idx=train_input.shape[1];
        elif dataset=="test":
          end_idx=test_input.shape[1];
      else:
        start_idx=set_idx[block];
        end_idx=set_idx[block+1];
      return start_idx, end_idx;
    
  all_train_states=np.array([]);
  for block in xrange(num_classes):
    start_idx, end_idx=calculate_block(tr_start_idx, block, num_classes);
      
    temp_train_states=network.drive_class(train_input[:, start_idx:end_idx]);
    if not all_train_states.size:
      all_train_states=temp_train_states;
    else:
      all_train_states=np.hstack((all_train_states,temp_train_states));
  print "[MESSAGE] Train data driven"
  
  x_test=network.drive_class(test_input);
  xTx=x_test.T.dot(x_test).diagonal();
  print "[MESSAGE] Test data driven"
    
  # setup for classification 2 classes
  curr_num_classes=2;
  _, curr_num_train=calculate_block(tr_start_idx, curr_num_classes-1, num_classes);
  _, curr_num_test=calculate_block(te_start_idx, curr_num_classes-1, num_classes, "test");
  
  print "[MESSAGE] Current number of classes: %d" % (curr_num_classes);
  
  ## init for correlation matrix list
  R_poss=[];
  for block in xrange(curr_num_classes):
    start_idx, end_idx=calculate_block(tr_start_idx, block, num_classes);
    
    states_c=all_train_states[:, start_idx:end_idx];
    R_poss.append(states_c.dot(states_c.T));
  
  ## compute positive conceptors for each class
  C_pos=[];
  for block in xrange(curr_num_classes):
    start_idx, end_idx=calculate_block(tr_start_idx, block, num_classes);
      
    c_pos=network.compute_pos_conceptor(R_poss[block],
                                        ap_pos,
                                        end_idx-start_idx);
    C_pos.append(c_pos);
      
    print "[MESSAGE] Best conceptors computed"
    
  pos_ev=np.zeros((curr_num_classes, curr_num_test));
  for i in xrange(curr_num_classes):
    for j in xrange(curr_num_test):
      pos_ev[i,j]=x_test[:,j].dot(C_pos[i]).dot(x_test[:,j][None].T)/xTx[j];
        
    print "[MESSAGE] %i class evidence is calculated" % (i+1);
  
  pos_out_label=np.argmax(pos_ev, axis=0);
  pos_accuracy=float(np.sum(pos_out_label==test_label[0:curr_num_test]))/float(curr_num_test);
  print "[MESSAGE] Accuracy %.2f %%" % (pos_accuracy*100);
  
  info=np.column_stack((curr_num_classes, pos_accuracy));
  np.savetxt(save_file, info, delimiter=',',newline='\n');
  
  while curr_num_classes<num_classes:
    curr_num_classes+=1;
    _, curr_num_train=calculate_block(tr_start_idx, curr_num_classes-1, num_classes);
    _, curr_num_test=calculate_block(te_start_idx, curr_num_classes-1, num_classes, "test");
    
    print "[MESSAGE] Current number of classes: %d" % (curr_num_classes);
    
    # get new R
    start_idx, end_idx=calculate_block(tr_start_idx, curr_num_classes-1, num_classes);
    states_c=all_train_states[:, start_idx:end_idx];
    R_poss.append(states_c.dot(states_c.T));
  
    # get new Conceptor
    c_pos=network.compute_pos_conceptor(R_poss[curr_num_classes-1],
                                        ap_pos,
                                        end_idx-start_idx);
    C_pos.append(c_pos);
    
    print "[MESSAGE] New conceptor is computed"
    
    pos_ev=np.zeros((curr_num_classes, curr_num_test));
    for i in xrange(curr_num_classes):
      for j in xrange(curr_num_test):
        pos_ev[i,j]=x_test[:,j].dot(C_pos[i]).dot(x_test[:,j][None].T)/xTx[j];
        
      print "[MESSAGE] %i class evidence is calculated" % (i+1);
  
    pos_out_label=np.argmax(pos_ev, axis=0);
    #print pos_out_label;
    #print test_label[0:curr_num_test];
    pos_accuracy=float(np.sum(pos_out_label==test_label[0:curr_num_test]))/float(curr_num_test);
    print "[MESSAGE] Accuracy %.2f %%" % (pos_accuracy*100);
    
    info=np.column_stack((curr_num_classes, pos_accuracy));
    np.savetxt(save_file, info, delimiter=',',newline='\n');
    
    del pos_ev;
  
  save_file.close();