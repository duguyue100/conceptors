"""
@author: Yuhuang Hu
@contact: duguyue100@gmail.com

@note: setup experiment of supervised classification task.
"""

import numpy as np;
from sacred import Experiment;

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

@exp.automain
def classify_experiment(filename_train,
                        filename_test,
                        save_path,
                        range_start,
                        range_end,
                        range_step):
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
  
  neuron_range=np.arange(range_start, range_end+range_step, range_step);
  
  print train_data.shape
  print test_data.shape;
  
  