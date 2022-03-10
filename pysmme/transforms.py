"""
This module contains various transforms for computing fast matrix vector products in specific situations.

"""

import numpy as np
from ._smme import WT, IWT

def wt(x, wf = "la8", J = None):
 r"""Discrete wavelet transform.

   This function performs a level J wavelet transform of the input array (1d, 2d, or 3d) 
   using the pyramid algorithm (Mallat 1989). Implemented in C by Brandon Whithcer.

   Parameters
   ----------
   x : np.array
       A 1, 2, or 3 dimensional data array. The size of each dimension must be dyadic.

   wf : string
       The type of wavelet family used.

   J : int
       J is the level (depth) of the decomposition. For default None the max
       depth is used and  ``wt(x)`` is equal to multiplying ``x`` with the
       corresponding wavelet matrix.

   Returns
   -------
   np.array
       np.array of shape identical to input x continant the transformed x

   Notes
   -----
   This is a C++/Python wrapper function for a C implementation of the
   discrete wavelet transform. Given a data array (1d, 2d or 3d) with dyadic
   dimensions sizes this transform is computed efficiently via the pyramid
   algorithm using C routines from  Brandon Whitcher's Waveslim package for R, see
   Percival and Walden (2000); Gencay, Selcuk and Whitcher (2001).

   This functionality is used in the computations underlying {{softmaximin}}
   to perform multiplications involving the wavelet (design) matrix efficiently.

   References
   ----------
   .. [3]  Gencay, R., F. Selcuk and B. Whitcher (2001) An Introduction to Wavelets and
      Other Filtering Methods in Finance and Economics, Academic Press.

   .. [4] Mallat, S. G. (1989) A theory for multiresolution signal decomposition: the
      wavelet representation, IEEE Transactions on Pattern Analysis and Machine
      Intelligence, 11, No. 7, 674-693.

   .. [5] Percival, D. B. and A. T. Walden (2000) Wavelet Methods for Time Series
       Analysis, Cambridge University Press.

   Examples
   --------
   >>> d = np.reshape(np.arange(1,2**3 + 1,1), (2, 2, 2), order = "F")
   >>> d1 = wt(d)
   >>> d2 = np.array([[[ 1.41421356e+00,  4.16333634e-17],
        [ 5.65685425e+00, -3.33644647e-16]],
       [[ 2.82842712e+00, -2.77555756e-17],
        [-2.64953102e-16,  1.27279221e+01]]])
   >>> iwt(d2)     
   """
 d = np.array(x.shape)
 dim = sum(d > 1)
 p1 = d[0]
 if(J == None):
  J = int(np.log2(min(d[d > 1])))
 
 p3 = 1 
 p2 = 1
 if(dim > 1):
  p2 = d[1]
 if(dim > 2):
  p3 = d[2]
 X = np.reshape(x, (p1, p2 * p3), order = "F")
 print("dd")
 out = WT(X, dim, wf, J, p1, p2, p3) #p1 x p2 * p3
 #out <- drop(array(out, c(p1, p2, p3)))
 #if(is.null(dim(out))){out <- as.matrix(out)}
 return np.reshape(out, (p1, p2, p3), order = "F")

def iwt(x, wf = "la8", J = None):
 r"""Discrete inverse wavelet transform.

  This function performs a level J wavelet transform of the input array (1d, 2d, or 3d) 
  using the pyramid algorithm (Mallat 1989). Implemented in C by Brandon Whithcer.

  Parameters
  ----------
  x : np.array
      a 1, 2, or 3 dimensional data array. The size of each dimension must be dyadic.

  wf : string
      the type of wavelet family used.

  J : int
      J is the level (depth) of the decomposition. For default None the max
      depth is used making  ``wt(x)`` equal to multiplying ``x`` with the
      corresponding wavelet matrix.

  Returns
  -------
  np.array
      np.array of shape identical to input x continant the transformed 

  Notes
  -----
  This is a C++/Python wrapper function for a C implementation of the
  discrete inverse wavelet transform. Given a data array (1d, 2d or 3d) with dyadic
  dimensions sizes this transform is computed efficiently via the pyramid
  algorithm using C routines from  Brandon Whitcher's Waveslim package for R, see
  Percival and Walden (2000); Gencay, Selcuk and Whitcher (2001).

  This functionality is used in the computations underlying ``softmaximin``
  to perform multiplications involving the wavelet (design) matrix efficiently.

  References
  ----------

  Examples
  --------

  """    
 d = np.array(x.shape)
 dim = sum(d > 1)
 p1 = d[0]
 if(J == None):
  J = int(np.log2(min(d[d > 1])))
 p3 = 1 
 p2 = 1
 if(dim > 1):
  p2 = d[1]
 if(dim > 2):
  p3 = d[2]
 X = np.reshape(x, (p1, p2 * p3), order = "F")
 out = IWT(X, dim, wf, J, p1, p2, p3) #p1 x p2 * p3
 #out <- drop(array(out, c(p1, p2, p3)))
 #if(is.null(dim(out))){out <- as.matrix(out)}
 return np.reshape(out, (p1, p2, p3), order = "F")

def H(M, A):
  d = A.shape
  Amat = np.reshape(A, [d[0], np.prod(d[1:len(d)])], "F")
  MAmat = np.matmul(M, Amat)
  newdim = list(d[1:len(d)])
  newdim.insert(0, MAmat.shape[0])
  return np.reshape(MAmat, newdim, "F")
  
# Rotation of an array A
def Rotate(A):
  d = np.arange(1, len(A.shape) + 1, 1)
  d[len(A.shape) - 1] = 0
  return np.transpose(A, d)

def RH(M, A):
  r"""The Rotated H-transform of a 3d Array by a Matrix.

  This function is an implementation of the :math:`\rho`-operator found in
  {Currie et al 2006}. It forms the basis of the GLAM arithmetic.

  Parameters
  ----------
  M : np.array
      A :math:`{n \times p_1}` matrix.

  A : np.array
      A 3d array of size :math:`p_1 \times p_2 \times p_3`.

  Returns
  -------
  np.array
      A 3d array of size :math:`p_2 \times p_3 \times n`.

  Notes
  -----
  For details see {Currie et al 2006} [2]. Note that this particular implementation
  is not used in the  routines underlying the optimization procedure.

  References
  ----------
  .. [2]  Currie, I. D., M. Durban, and P. H. C. Eilers (2006). Generalized linear
     array models with applications to multidimensional smoothing.
     {Journal of the Royal Statistical Society. Series B}. 68, 
     259-280. url = {http://dx.doi.org/10.1111/j.1467-9868.2006.00543.x}.

  Examples
  --------
  
  >>> n1 = 15; n2 = 4; n3 = 3; p1 = 12; p2 = 3; p3 = 4
  >>> ###marginal design matrices (Kronecker components)
  >>> X1 = np.random.normal(0, 1, (n1, p1))
  >>> X2 = np.random.normal(0, 1, (n2, p2))
  >>> X3 = np.random.normal(0, 1, (n3, p3))
  >>> A = np.random.normal(0, 1, (p1, p2, p3))
  >>> R1 = RH(X3, RH(X2, RH(X1, A)))
  >>> R2 = np.matmul(np.kron(X3, np.kron(X2, X1)), np.reshape(A, [p1 * p2 * p3, 1], "F"))
  >>> max(abs(np.reshape(R1, [n1 * n2 * n3, 1], "F") - R2)) 
  """    
  return Rotate(H(M, A))

