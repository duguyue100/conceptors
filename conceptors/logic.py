'''
Created on Mar 5, 2015

@author: Yuhuang Hu
@note: This module contains logical operations over conceptors
'''

import numpy as np;

def NOT(R, out_mode="simple"):
  """
  Compute NOT operation of conceptor.
  
  @param R: covariance matrix of conceptor.
  @param out_mode: output mode ("simple"/"complete")
  
  @return not_R: negative of R
  @return U: eigen vectors of not_R
  @return S: eigen values of not_R
  """
  
  dim=R.shape[0];
  
  not_R=np.eye(dim)-R;
  
  if out_mode=="simple":
    return not_R;
  elif out_mode=="complete":
    U, S, _=np.linalg.svd(not_R);
    return not_R, U, S;
  else:
    return not_R; # should throw a error message here actually
  
def AND(R, Q, tol=1e-14, out_mode="simple"):
  """
  Compute AND Operation of two conceptors
  
  @param R: covariance matrix of a conceptor
  @param Q: covariance matrix of a conceptor
  @param tol: 
  @param out_mode: output mode ("simple"/"complete")
  """
  
  dim=R.shape[0];
  
  ur, sr, _=np.linalg.svd(R);
  uq, sq, _=np.linalg.svd(Q);
  
  num_rank_r=np.sum((sr>tol).astype(float));
  num_rank_q=np.sum((sq>tol).astype(float));
  
  uro=ur[:, num_rank_r+1:];
  uqo=uq[:, num_rank_q+1:];
  
  W, sigma, _=np.linalg.svd(uro.dot(uro.T)+uqo.dot(uqo.T));
  num_rank_sigma=np.sum((sigma>tol).astype(float));
  Wgk=W[:, num_rank_sigma+1:];
  
  r_and_q=Wgk.dot(np.linalg.inv(Wgk.T.dot(np.linalg.pinv(R, tol)+np.linalg.pinv(Q, tol)-np.eye(dim)).dot(Wgk))).dot(Wgk.T);
  
  if out_mode=="simple":
    return r_and_q;
  elif out_mode=="complete":
    u,s,_=np.linalg.svd(r_and_q);
    return r_and_q, u, s;
  else:
    return r_and_q;
  

def OR(R, Q, out_mode="simple"):
  """
  Compute OR operation of two conceptors
  
  @param R: covariance matrix of a conceptor
  @param Q: covariance matrix of a conceptor
  @param out_mode: output mode ("simple"/"complete")
  
  @return not_R: negative of R
  @return U: eigen vectors of not_R
  @return S: eigen values of not_R
  """
  
  R_or_Q=NOT(AND(NOT(R), NOT(Q)));

  if out_mode=="simple":
    return R_or_Q;
  elif out_mode=="complete":
    U, S, _=np.linalg.svd(R_or_Q);
    return R_or_Q, U, S;
  else:
    return R_or_Q;
  
  
def PHI(C, gamma):
  """
  Aperture adaptation
  
  @param C: conceptor matrix
  @param gamma: adaptation parameter
  
  @return C_new: updated new conceptor matrix
  """
  
  dim=C.shape[0];
  
  if gamma==0:
    U,S,_=np.linalg.svd(C);
    S[S<1]=np.zeros((np.sum((S<1).astype(float)), 1));
    C_new=U.dot(S).dot(U.T);
  elif gamma==np.Inf:
    U,S,_=np.linalg.svd(C);
    S[S>0]=np.zeros((np.sum((S>0).astype(float)), 1));
    C_new=U.dot(S).dot(U.T);
  else:
    C_new=C.dot(np.linalg.inv(C+gamma**-2*(np.eye(dim)-C)));
    
  return C_new;

def conceptor_NOT(C):
  """
  Logic NOT
  
  @param C: a conceptor
  """
  
  return np.eye(C.shape[0])-C;

def conceptor_AND(C, B):
  """
  Logic AND, compute C and B
  
  @param C: a conceptor
  @param B: a conceptor
  """
  
  return np.linalg.inv(np.linalg.inv(C)+np.linalg.inv(B)-np.eye(C.shape[0]));
  
def conceptor_OR(C, B):
  """
  Logic OR, compute C or B
  
  @param C: a conceptor
  @param B: a conceptor
  """
  
  #return np.linalg.inv(I+np.linalg.inv(C.dot(np.linalg.inv(I-C))+B.dot(np.linalg.inv(I-B))));
  return conceptor_NOT(conceptor_AND(conceptor_NOT(C), conceptor_NOT(B)));