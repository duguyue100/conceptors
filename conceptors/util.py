'''
Created on Feb 25, 2015

@author: Yuhuang Hu
@note: useful utilities functions for assisting conceptor networks
'''

import os;
import cPickle as pickle
import numpy as np;
import numpy.matlib;
import scipy.sparse;
import scipy.sparse.linalg;
import matplotlib.pyplot as plt;

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
  
def read_jpv_data(train_file,
                  test_file):
  """
  Function of reading Japanese Vouwels dataset
  
  There are 270 recordings in training file, and 370 recordings in testing file
  
  @param train_file: string of training file location
  @param test_file: string of testing file location
  
  @return 
  """
  
  num_train=270;
  num_test=370;
  train=np.loadtxt(train_file);
  test=np.loadtxt(test_file);
  
  train_inputs=[];
  read_index=0;
  for c in xrange(num_train):
    l=0;
    while (train[read_index, 0]!=1.0):
      l+=1;
      read_index+=1;
      
    train_inputs.append(np.hstack((train[read_index-l:read_index,:],
                                   np.linspace(1.0/l, 1, l)[None].T,
                                   (l/30.0)*np.ones((l,1)))));
    read_index+=1;
                                   
  test_inputs=[];
  read_index=0;
  for c in xrange(num_test):
    l=0;
    while (test[read_index, 0]!=1.0):
      l+=1;
      read_index+=1;
      
    test_inputs.append(np.hstack((test[read_index-l:read_index,:],
                                  np.linspace(1.0/l, 1, l)[None].T,
                                  (l/30.0)*np.ones((l,1)))));
    read_index+=1;
    
  train_outputs=[];
  for c in xrange(num_train):
    l=train_inputs[c].shape[0];
    teacher=np.zeros((l,9));
    speaker_index=np.ceil(c/30);
    teacher[:, speaker_index]=np.ones((l,1))[:,0];
    train_outputs.append(teacher);
  
  test_outputs=[];
  speaker_index=0;
  block_counter=0;
  block_lengthes=np.asarray([31, 35, 88, 44, 29, 24, 40, 50, 29]);
  
  for c in xrange(num_test):
    block_counter+=1;
    
    if block_counter==block_lengthes[speaker_index]+1:
      speaker_index+=1;
      block_counter=1;
      
    l=test_inputs[c].shape[0];
    teacher=np.zeros((l,9));
    teacher[:, speaker_index]=np.ones((l,1))[:,0];
    test_outputs.append(teacher);
  
  return train_inputs, train_outputs, test_inputs, test_outputs;

def normalize_jap_data(data):
  """
  Normalize Japanese Vouwels data
  
  @param data: input data list
  
  @return: 
  """
  
  num_data=len(data);
  total_length=sum([d.shape[0] for d in data]);
  
  all_data=np.zeros((total_length, 12));
  curr_start_index=0;
  
  for s in xrange(num_data):
    l=data[s].shape[0];
    all_data[curr_start_index:curr_start_index+l,:]=data[s][:,0:12];
    curr_start_index+=l;
    
  max_vals=np.max(all_data,0);
  min_vals=np.min(all_data,0);
  shifts=-min_vals;
  scales=1./(max_vals-min_vals);
  norm_data=[];
  
  for s in xrange(num_data):
    d=data[s][:,0:12]+numpy.matlib.repmat(shifts, data[s].shape[0], 1);
    d=d.dot(np.diag(scales));
    norm_data.append(np.hstack((d,
                                data[s][:,12:14])));
  
  return norm_data, shifts, scales;

def transform_jap_data(data,
                       shifts,
                       scales):
  """
  Transform Japanese Vouwels data to standard form
  
  @param data: input data list
  @param shifts: shifts of the data
  @param scales: scales of the data
  
  @return: 
  """

  num_data=len(data);
  
  trans_data=[];
  
  for s in xrange(num_data):
    d=data[s][:,0:12]+numpy.matlib.repmat(shifts, data[s].shape[0], 1);
    d=d.dot(np.diag(scales));
    trans_data.append(np.hstack((d,
                                 data[s][:,12:14])));
    
  return trans_data;

def load_CIFAR_batch(filename):
    """
    load single batch of cifar-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param filename: string of file name in cifar
    @return: X, Y: data and labels of images in the cifar batch
    """
    
    with open(filename, 'r') as f:
        datadict=pickle.load(f);
        
        X=datadict['data'];
        Y=datadict['labels'];
        
        X=X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float");
        Y=np.array(Y);
        
        return X, Y;
        
        
def load_CIFAR10(ROOT):
    """
    load entire CIFAR-10 dataset
    
    code is adapted from CS231n assignment kit
    
    @param ROOT: string of data folder
    @return: Xtr, Ytr: training data and labels
    @return: Xte, Yte: testing data and labels
    """
    
    xs=[];
    ys=[];
    
    for b in range(1,6):
        f=os.path.join(ROOT, "data_batch_%d" % (b, ));
        X, Y=load_CIFAR_batch(f);
        xs.append(X);
        ys.append(Y);
        
    Xtr=np.concatenate(xs);
    Ytr=np.concatenate(ys);
    
    del X, Y;
    
    Xte, Yte=load_CIFAR_batch(os.path.join(ROOT, "test_batch"));
    
    return Xtr, Ytr, Xte, Yte;

def visualize_CIFAR(X_train,
                    y_train,
                    samples_per_class):
    """
    A visualize function for CIFAR 
    """
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    num_classes=len(classes);
    
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    
    plt.show();








