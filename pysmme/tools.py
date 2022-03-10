"""
This module contains functionality for i) solving (fitting, calibrating) the soft maximin problem 
and ii) predicting from this fitted solution (model).

"""
import numpy as np
from ._smme import pga
from pysmme.transforms import iwt, RH

class _smme_dict(dict):
   def __init__(self, d):
      self._dict = d      
   def __repr__(self):
      out = self._dict["spec"] + ":" + "\n"
      j = 0
      for z in self._dict["zeta"]: # list of length num of z of np arrays of len endmod(z)
        out = out + "\n" + "{:<15} {:<8} {:<8}".format("zeta = " + str(z), "DF", "Lambda")
        for i in range(len(self._dict["df"][j])):
          df = self._dict["df"][j][i]
          lamb = self._dict["lamb"][j][i]
          out = out + "\n" + "{:<15} {:<8} {:<8}".format("  ", df, lamb)
        j = j + 1
      return out
      
def softmaximin(
y,#, #G list or d array
x, #G list or d list or string for wavelets
zeta,
penalty,
alg,
nlamb = 30,
lamb_min_ratio = 1e-04,
lamb = None,
scale_y = 1,
penalty_factor = None,
reltol = 1e-05,
maxiter = 1000,
steps = 1,
btmax = 100,
c = 0.0001,
tau = 2,
M = 4,
nu = 1,
Lmin = 0,
lse = True,
nthreads = 4):
  
  r"""Efficient procedure for solving the Lasso or SCAD penalized soft maximin problem.

  This software implements two proximal
  gradient based algorithms (NPG and FISTA) to solve different forms of the soft
  maximin problem from Lund et al., 2022 see [1]. 1) For general group specific
  design the soft maximin problem is solved using the NPG algorithm.
  2) For fixed identical design across groups, the estimation procedure uses 
  either the FISTA algorithm or the NPG algorithm in the following two cases:
  i) For a tensor design matrix the algorithms use array arithmetic  to 
  avoid the design matrix and speed computations ii) For a wavelet based design 
  matrix the algorithms use the pyramid algorithm to avoid the design matrix and speed up
  computations.

  Multi-threading is possible when openMP is available.

  Parameters
  ----------
  y : list of arrays or array
    For a model with varying design across groups a list containing the :math:`G` group specific 
    response vectors of sizes :math:`n_i \times 1` . For a model with identical design 
    across :math:`G` groups, 
    an array of size :math:`n_1 \times\cdots\times n_d \times G` (:math:`d \in \{ 1, 2, 3\}`).
  x : list of arrays or string
    For a model with varying design across groups a list containing the 
    :math:`G` group specific design matrices of sizes :math:`n_i \times p_i`.  
    For a model with identical design across :math:`G` groups, either i) a list containing the 
    :math:`d \in \{ 1, 2, 3\}` marginal design matrices (tensor components) or ii) 
    a string indicating the type of wavelets to be used, see ``pysmme.transforms.wt`` for options. 
  zeta : array of strictly positive floats 
    controls  the soft maximin approximation accuracy. When ``len(zeta) > 1`` 
    the procedure will distribute
    the computations using the   ``nthreads`` parameter below when openMP is available.
  penalty : string 
    specifies the penalty type. Possible values are ``lasso, scad``.
  alg : string 
    specifies the optimization algorithm. Possible values are ``npg, fista``.
  nlambda : strictly positive int 
    The number of ``lamb`` values used when lamb is not specified.
  lamb_min_ratio : strictly positive float 
    controls minimum ``lamb`` values by setting the ratio bewtween 
    :math:`\lambda_{max}` -- the (data dependent) smallest value for which all 
    coefficients are zero --  and the smallest value for ``lamb``.
    Used when lamb is not specified.
  lamb : array of strictly positive floats 
    used  as penalty parameters.
  scale_y : strictly positive float 
    that is the response   ``y`` is multiplied with.
  penalty_factor : array 
    size :math:`p_1 \times \cdots \times p_d` of positive floats. 
    Is multiplied with each element in   ``lamb`` to allow
    differential penalization on the coefficients. 
  reltol : strictly positive float 
    giving the convergence tolerance for the inner loop.
  maxiter : positive int
    giving the maximum number of  iterations
    allowed for each   ``lamb`` value, when  summing over all outer iterations
    for said   ``lamb``.
  steps : strictly positive int 
    giving the number of steps used in the
    multi-step adaptive lasso algorithm for non-convex penalties. Automatically
    set to 1 when   ``penalty = "lasso"``.
  btmax : strictly positive integer giving the maximum number of backtracking
    steps allowed in each iteration. 
  c : strictly positive float 
    used in the NPG algorithm. 
  tau : strictly positive float 
    used to control the stepsize for NPG. 
  M : pos int
    giving the look back for the NPG. 
  nu : strictly positive 
    float used to control the stepsize in the proximal algorithm. A  value less than 1 will decrease 
    the stepsize and a value larger than one will increase it.
  Lmin : pos float 
    used by the NPG algorithm to control the
    stepsize. For the default  ``Lmin = 0`` the maximum step size is the same
    as for the FISTA algorithm.
  lse : bool 
    indicating whether to use log sum exp-loss.  TRUE is
    default and yields the loss below.
  nthreads : pos int
    giving the number of threads to use when  openMP  is available. 

  Returns
  -------  
  spec : string 
   contains specifications of the model fitted by the function call
  coef : array
   A math:`p \times`nlamb matrix containing the
   estimates of the model coefficients (  :math:`beta`) for each   ``lamb``-value
   for which the procedure converged. When   ``len(zeta) > 1``
   a   ``len(zeta)``-list of such matrices.
  lamb : Array
   containing the sequence of penalty values used
   in the estimation procedure for which the procedure converged.
   When   ``len(zeta) > 1`` a   ``len(zeta)``-list of such vectors.
  Obj : array
   containing the objective values for each
   iteration and each model for which the procedure converged.
   When   ``len(zeta) > 1`` a   ``len(zeta)``-list of such matrices.
  df : array 
   Indicating the nonzero model coefficients for each
   value of   ``lamb`` for which the procedure converged. When
   ``len(zeta) > 1`` a   ``len(zeta)``-list of such vectors.
  dimcoef : int or np.array
   Indicating the number :math:`p` of model parameters.
   For array data a vector giving the dimension of the model coefficient array :math:`\beta`
  dimobs : integer
    The number of observations. For array data a vector giving the number of  observations in each dimension.
  dimmodel : int
   The dimension of the array model. ``None`` for general models.
  iter : np.array 
   The  number of  iterations for each ``lamb`` value for which the procedure converged. When
   ``len(zeta) > 1`` a   ``len(zeta)``-list of such vectors. ``bt_iter``  is total 
   number of backtracking steps performed,
   ``bt_enter`` is the number of times the backtracking is initiated,
   and   ``iter`` is a vector containing the  number of  iterations for each
   ``lamb`` value and    ``iter`` is total number of iterations. TODO!!! notoutputted

  Notes
  -----
  Consider modeling heterogeneous data :math:`\{y_1,\ldots, y_n\}` by dividing
  it into :math:`G` groups :math:`\mathbf{y}_g = (y_1, \ldots, y_{n_g})` ,
  :math:`g \in \{ 1,\ldots, G\}` and then using a linear model
  
  .. math:: \mathbf{y}_g = \mathbf{X}_gb_g + \epsilon_g, \  g \in \{1,\ldots, G\},
  
  to model the group response. Then :math:`b_g` is a group specific :math:`p\times 1`
  coefficient vector, :math:`\mathbf{X}_g` an :math:`n_g\times p` group design matrix and
  :math:`\epsilon_g` an :math:`n_g\times 1` error term. The objective is to estimate
  a common coefficient :math:`\beta` such that :math:`\mathbf{X}_g\beta` is a robust
  and good approximation to :math:`\mathbf{X}_gb_g` across groups.
 
  Following [1], this objective may be accomplished by
  solving the soft maximin estimation problem
  
  .. math:: \min_{\beta}\frac{1}{\zeta}\log\bigg(\sum_{g = 1}^G \exp(-\zeta \hat V_g(\beta))\bigg) + \lambda  \Vert\beta\Vert_1, \quad \zeta > 0,\lambda \geq 0.
  
  Here :math:`\zeta` essentially controls the amount of pooling across groups
  (:math:`\zeta \sim 0` effectively ignores grouping and pools observations) and
   
  .. math:: \hat V_g(\beta):=\frac{1}{n_g}(2\beta^\top \mathbf{X}_g^\top \mathbf{y}_g -\beta^\top \mathbf{X}_g^\top \mathbf{X}_g\beta),
  
  is the empirical explained variance, see [2] for more
  details and references.
 
  The function  ``softmaximin`` solves the soft maximin estimation problem in
  large scale settings for a sequence of penalty parameters
  :math:`\lambda_{max}>\ldots >\lambda_{min}>0` and a sequence of strictly positive
  softmaximin  parameters :math:`\zeta_1, \zeta_2,\ldots`.
 
  The implementation also solves the
  problem above with the penalty given by the SCAD penalty, using the multiple
  step adaptive lasso procedure to loop over the inner proximal algorithm.
 
  Two optimization algorithms  are implemented in the SMME packages;
  a non-monotone proximal gradient (NPG) algorithm and a fast iterative soft
  thresholding algorithm (FISTA).
 
  The implementation is particularly efficient for models where the design is
  identical across groups i.e. :math:`\mathbf{X}_g = \mathbf{X}`
  :math:`\forall g \in \{1, \ldots, G\}` in the following two cases:

  i) first if :math:`\mathbf{X}` has Kronecker (tensor) structure i.e. for marginal :math:`n_i\times p_i` design matrices :math:`\mathbf{M}_1,\ldots, \mathbf{M}_d`
  , :math:`d \in \{ 1, 2, 3\}`,

  .. math:: \mathbf{X} = \bigotimes_{i=1}^d \mathbf{M}_i 
  
  then ``y`` is a :math:`d + 1` dimensional response array
  and    ``x`` is a list containing the :math:`d` marginal matrices
  :math:`\mathbf{M}_1,\ldots, \mathbf{M}_d`. In this case  softmaximin solves
  the soft maximin problem using minimal memory by way of tensor optimized
  arithmetic, see also   ``RH``.

  ii) second, if the design matrix :math:`\mathbf{X}` is the inverse matrix of an
  orthogonal wavelet transform  then ``softmaximin``  will solve the soft maximin 
  problem given  ``x = str`` -- where ``str`` is a shorthand for the wavelet basis (see....) -- and 
  the :math:`d + 1` dimensional response array  ``y``. In this case  the  pyramid algorithm is used 
  to compute multiplications involving :math:`\mathbf{X}`.
 
  Note that when multiple values for :math:`\zeta` is provided it is  possible to
  distribute the computations across CPUs if openMP is available.
 
  References
  ----------
  .. [1] Lund, A., S. W. Mogensen and N. R. Hansen (2022). Soft Maximin Estimation for
      Heterogeneous Data. Scandinavian Journal of Statistics. url = {https://doi.org/10.1111/sjos.12580}
 
  Examples
  --------
  #Non-array data ##size of example

  >>> G = 3; 
  >>> n = np.array([65, 26, 13])
  >>> p = np.array([13, 5, 4])
  
  ##marginal design matrices (Kronecker components)

  >>> x = [None] * 3 
  >>> for i in range(len(x)):
  >>> x[i] = np.random.normal(0, 1, (n[i], p[i]))
       
  ##common features and effects

  >>> common_features = np.random.binomial(1, 0.1, np.prod(p)) #sparsity of common effects
  >>> common_effects = np.random.normal(size = np.prod(p)) * common_features
  
  ##group response

  >>> y = np.zeros((n[0], n[1], n[2], G))
  >>> for g in range(G):
  >>>     bg = np.random.normal(0, 0.1, np.prod(p)) * (1 - common_features) + common_effects
  >>>     mu = RH(x[2], RH(x[1], RH(x[0], np.reshape(bg, (p[0], p[1], p[2]), "F") )))
  >>>     y[:, :, :, g] = np.random.normal(0, 1, (n)) + mu
  
  ##fit model for range of lambda and zeta
  
  >>> zeta = np.array([0.1, 1, 10, 100])
  >>> fit = softmaximin(y, x, zeta = zeta, penalty = "lasso", alg = "npg")
  >>> modelno = 10
  >>> zetano = 2
  >>> betahat = fit["coef"][zetano][:, modelno]
   
  >>> f, ax = plt.subplots(1)
  >>> ax.plot(common_effects, "r+")
  >>> ax.plot(betahat)
  >>> plt.show() 
   
  #Array data and wavelets
  ##size of example

  >>> set.seed(42)
  >>> G = 5; 
  >>> p = n = np.array([2**2, 2**3, 2**4])
  
  ##common features and effects
  
  >>> common_features = np.random.binomial(1, 0.1, np.prod(p)) #sparsity of common effects
  >>> common_effects = np.random.normal(size = np.prod(p)) * common_features
  
  ##group response

  >>> y = np.zeros((n[0], n[1], n[2], G))
  >>> for g in range(G):
  >>>     bg = np.random.normal(0, 0.1, np.prod(p)) * (1 - common_features) + common_effects
  >>>     mu = iwt(np.reshape(bg, (p[0], p[1], p[2]), "F"))
  >>>     y[:, :, :, g] = np.random.normal(0, 1, (n)) + mu
  
  ##fit model for range of lambda and zeta
  
  >>> zeta = np.array([0.1, 1, 10, 100])
  >>> fit = softmaximin(y, x, zeta = zeta, penalty = "lasso", alg = "npg")
  >>> modelno = 10
  >>> zetano = 2
  >>> betahat = fit["coef"][zetano][:, modelno]
   
  >>> f, ax = plt.subplots(1)
  >>> ax.plot(common_effects, "r+")
  >>> ax.plot(betahat)
  >>> plt.show() 
  
  ##Non-array data
  ##size of example

  >>> G = 10
  >>> n = np.random.choice(np.arange(100,500,1), G) #sample(100:500, G); 
  >>> p = 60
  >>> x = [None] * G
  
  ##group design matrices

  >>> for i in range(len(x)):
  >>> x[i] = np.random.normal(0, 1, (n[i], p))
  
  ##common features and effects

  >>> common_features = np.random.binomial(1, 0.1, np.prod(p)) #sparsity of common effects
  >>> common_effects = np.random.normal(size = np.prod(p)) * common_features
  
  ##group response

  >>> y = [None] * G
  >>> for g in range(G):
  >>>     bg = np.random.normal(0, 0.5, np.prod(p)) * (1 - common_features) + common_effects
  >>>     mu = np.matmul(x[g], bg)
  >>>     y[g] = np.random.normal(0, 1, n[g]) + mu
  
  ##fit model for range of lamb and zeta
  
  >>> fit = softmaximin(y, x, zeta = zeta, penalty = "lasso", alg = "npg")
  >>> betahat = fit["coef"]
  
  ##estimated common effects for specific lamb and zeta

  >>> modelno = 6 
  >>> zetano = 2
  >>> f, ax = plt.subplots(1)
  >>> ax.plot(common_effects, "r+")
  >>> ax.plot(betahat[zetano][:, modelno])
  >>> plt.show() 

  """

  try:
   num_of_zeta = len(zeta)
  except TypeError:
   num_of_zeta = 1
    
  Z = [] 
  if(type(y) == np.ndarray): #array data fixed design
    y = y * scale_y
    fix_array = True 
    G = y.shape[len(y.shape) - 1]
    dimglam = len(y.shape) - 1 #
    if (dimglam not in  [1, 2, 3]):
      return "Error: The dimension of an array model must be d = 1, 2 or 3!"
    if(type(x) ==  str): #wavelet design
      if(x not in ["la8"]): ##todo!!!!
        return "Error: The wavelet design is not correctly specified"
      if(np.mean(np.round(np.log2(y.shape[0:dimglam])) == np.log2(y.shape[0:dimglam])) != 1):
        return "Error: data is not dyadic so cannot use wavelet design"
      wave = True
      wf = x
      p1 = n1 = y.shape[0]
      if(dimglam == 1):
        p2 = n2 = 1
        p3 = n3 = 1
        J = int(np.log2(p1))
      elif(dimglam == 2):
        p2 = n2 = y.shape[1]
        p3 = n3 = 1
        J = int(np.log2(min(p1, p2)))
      else:
        p2 = n2 = y.shape[1]
        p3 = n3 = y.shape[2]
        J = int(np.log2(min(p1, p2, p3)))
      p = n = n1 * n2 * n3
      #x = [np.ones((2, 2)) * n1, np.ones((2, 2)) * n2, np.ones((2, 2)) * n3]
      x = [n1, n2, n3]
    elif(type(x) == list): ##custom fixed array tensor design
      if(len(x) != dimglam): 
        return "Error: For array data y the number of marginal design matrices in x must equal the dimension of the group data (d = 1, 2, or 3)"
      J = 0
      wave = False
      wf = "whatever"
      if (dimglam == 1):
        x.append(np.ones((1, 1)))
        x.append(np.ones((1, 1)))
      if (dimglam == 2):
        x.append(np.ones((1, 1))) 
      n1 = x[0].shape[0]
      n2 = x[1].shape[0]
      n3 = x[2].shape[0]
      p1 = x[0].shape[1]
      p2 = x[1].shape[1]
      p3 = x[2].shape[1]
      n = n1 * n2 * n3
      p = p1 * p2 * p3
    else:
      return "Error: x not correctly specified"
   #structure array as G-list of n1xn2n3 matrices
    for i in range(G): 
      if(dimglam == 1):
        Z.append(np.reshape(y[:, i], (n1, n2 * n3) , order  = "F"))
      elif(dimglam == 2):
        Z.append(np.reshape(y[:,:, i], (n1, n2 * n3) , order  = "F"))
      elif(dimglam == 3):
        Z.append(np.reshape(y[:,:,:, i], (n1, n2 * n3) , order  = "F")) 
    del y  #is Z a copy?   
    if(penalty_factor == None):
      penalty_factor = np.ones((p1, p2 * p3), order  = "F")
    else:
      penalty_factor = np.reshape(penalty_factor, (p1, p2 * p3), order  = "F") 
  elif(type(y) == list): #varying design
    G = len(y)
    for i in range(G):
      Z.append(y[i] * scale_y)
    del y    
    fix_array = False
    wave = False
    dimglam = 0
    J = 0
    wf = "whatever"
    if(len(x) != G):
      return "Error: The number of group design matrices in x must equal the number of the groups G in the data"
    #check to make sure y is compaitble with x in every gropu...todo
    n = 0
    for i in range(G):
      n = n + Z[i].shape[0]
    p = x[0].shape[1]
    if(penalty_factor == None):
      penalty_factor = np.ones(p) #np
  else:
    return "Error: y must be list or array"
  if(alg not in ["npg", "fista"]):
    return "Error: algorithm must be correctly specified"
  if(c <= 0):
    return "Error: c must be strictly positive"
  if(Lmin < 0):
    return "Error: Lmin must be positive"
  #if(np.mean(zeta <= 0) > 0): todo!!
  #  return "all zetas must be strictly positive"
  if(penalty not in ["lasso", "scad"]):
    return "Error: penalty must be correctly specified"
  if(np.min(penalty_factor) < 0):
    return "Error: penalty.factor must be positive"
  if(penalty_factor.size != p):
    return "Error: number of elements in penalty.factor " + str(penalty_factor.size) + " is not equal to the number of coefficients " +  str(p)
  if(penalty == "lasso"):
    steps = 1
  if(lamb == None):
    makelamb = True
    lamb = np.zeros(nlamb) #fill with nas??
  else:
    lamb = scale_y * lamb
    makelamb = False
    nlamb = len(lamb)
   
  Coef, DF, Btiter, ITER, endmodelno, Lamb, Stops, openmp = pga(x,
                                                            Z,
                                                            penalty,
                                                            zeta,
                                                            c,
                                                            lamb, nlamb, makelamb, lamb_min_ratio,
                                                            penalty_factor,
                                                            reltol,
                                                            maxiter,
                                                            steps,
                                                            btmax,
                                                            M,
                                                            tau,
                                                            nu,
                                                            alg,
                                                            fix_array,
                                                            lse,
                                                            Lmin,
                                                            nthreads          ,
                                                            wave,
                                                            J,
                                                            dimglam,
                                                            wf)


  """ if(mean(res$Stops[2, ]) > 0):
    zs = which(res$Stops[2, ] != 0)

  warning(paste("maximum number of inner iterations (",maxiter,") reached for model no.",
                paste(endmodelno[zs] + 1, collapse = " ")," for zeta(s)",
                paste(zeta[zs], collapse = " ")))

  if(mean(res$Stops[3, ]) > 0):
    zs = which(res$Stops[3, ] != 0)

  warning(paste("maximum number of backtraking steps reached for model no.",
                paste(endmodelno[zs] + 1, collapse = " ")," for zeta(s)",
                paste(zeta[zs], collapse = " ")))

  if(res$openMP == 1):
    message(paste("Multithreading enabled using", nthreads, "threads"))
 
  # Iter = res$Iter
  #
  # maxiterpossible = sum(Iter > 0)
  # maxiterreached = sum(Iter >= (maxiter - 1))
  #
  # if(maxiterreached > 0){
  #
  # warning(
  # paste("maximum number of inner iterations (",maxiter,") reached ",maxiterreached," time(s) out of ",maxiterpossible," possible")
  # )
  #
  #}"""

  
  if(num_of_zeta > 1):
    #Obj = []
    iter = [None] *  num_of_zeta
    coef = [None] *  num_of_zeta
    lamb_out = [None] *  num_of_zeta
    df = [None] *  num_of_zeta
    for z in range(num_of_zeta):
      coef[z] = Coef[: , 0:int(endmodelno[z]) + 1, z] / scale_y
      lamb_out[z] = Lamb[0:int(endmodelno[z]) + 1, z] / scale_y
      df[z] = DF[0:int(endmodelno[z]) + 1, z]
      iter[z] = ITER[1:int(endmodelno[z]) + 1, z]
     ## Obj[[z]] = res$Obj[, 1:int(endmodelno[z]) ,1]
  else:
    coef = Coef[: , 0:int(endmodelno) + 1, 0] / scale_y
    lamb_out = Lamb[0:int(endmodelno) + 1, 0] / scale_y
    df = DF[0:int(endmodelno) + 1, 0]
    iter = ITER[0:int(endmodelno) + 1, 0]
   ## Obj = OBJ[:, 0:int(endmodelno) ,0]

  if(fix_array):
    if(wave == 0):
      spec = str(penalty) + "-penalized smme model with " + str(G) + " groups and fixed " + str(dimglam) + "D-tensor " + "custom "  + "design"
    else:
      spec = str(penalty) + "-penalized smme model with " + str(G) + " groups and fixed " + str(dimglam) + "D-tensor " + str(wf) + "-wavelet "  + "design"
  else:
    spec = str(penalty) + "-penalized smme model with varying inputs across the " + str(G) + " groups" 
   
  out = _smme_dict({"df":df, "lamb":lamb_out, "zeta":zeta, "spec":spec})
  out["spec"] = spec  
  out["zeta"] = zeta
  out["coef"] = coef
  out["lamb"] = lamb_out
  out["df"] = df
  out["iter"] = iter
  #out["Obj"] = Obj

  if(fix_array == 1):
    out["dimcoef"] = [p1, p2, p3]#[0:dimglam]
    out["dimobs"] = [n1, n2, n3]#[0:dimglam]
    out["dimmodel"] = dimglam
  else:
    out["dimcoef"] = p
    out["dimobs"] = n
    out["dimmodel"] = None

  out["endmod"] = endmodelno
  #out$BT = drop(res$BT)
  #Iter = list()
  ##Iter$bt_enter = res$btenter #vector   nzeta todo!
  #Iter$bt_iter = res$btiter #vector   nzeta
 # #Iter$sum_iter = sum(Iter$iter, na.rm = TRUE) #vector   nzeta todo iter!!
 # out["Iter"] = drop(Iter)
 # out["Stops"] = res$Stops

  return out

def predict(fit, x):

 r""" Make Prediction From an smme Object.
 
   Parameters
   ----------
   fit : smme_dict
       The output from a ``pysmme.tools.softmaximin`` call
 
   x : list, matrix or string
       an object that should be like the input to the ``pysmme.tools.softmaximin`` call that 
       produced the object ``fit``. For general  models a matrix
       with column dimension equal to that of  the original input. 
       For array models with custom design a list 
       and with wavelet design the name of the wavelet used. 
 
   Returns
   -------
   list     
        A list of length ``len(zeta)``. If ``x`` is an :math:`k \times p` matrix ``x``
        each list item is an :math:`k \times m_\zeta` matrix containing the linear
        predictors computed for each model. If ``x`` is a string or  a
        list of matrices each of size :math:`k_{i} \times p_i`,  each list item is an array
        of size :math:`k_1 \times \cdots \times k_d \times m_\zeta`,
        :math:`d\in \{1,2,3\}`, with the linear predictors computed for each model.
 
   Notes
   -----
    Given input ``x`` data this function computes the linear predictors
    using the fitted model coefficients supplied in  the ``object`` which should be produced by  
    ``softmaximin``. If ``object`` is the result of fitting general type model  
    ``x`` should be a :math:`k \times p` matrix (:math:`p` is the number of model
    coefficients and :math:`k` is the number of new data points). 
    If ``object`` is the result of fitting a model with tensor design `x`` should be a list containing 
    :math:`k_i \times p_i, i = 1, 2, 3` matrices 
    (:math:`k_i` is the number of new marginal data points in the :math:`i`th dimension).
    or a string indicating the wavelet used to produec ``object``.
   
   Examples
   --------
   
  #array data 
  ##size of example

  >>> G = 3; 
  >>> n = np.array([65, 26, 13])
  >>> p = np.array([13, 5, 4])
  
  ##marginal design matrices (Kronecker components)

  >>> x = [None] * 3 
  >>> for i in range(len(x)):
  >>> x[i] = np.random.normal(0, 1, (n[i], p[i]))
       
  ##common features and effects

  >>> common_features = np.random.binomial(1, 0.1, np.prod(p)) #sparsity of common effects
  >>> common_effects = np.random.normal(size = np.prod(p)) * common_features
  
  ##group response

  >>> y = np.zeros((n[0], n[1], n[2], G))
  >>> for g in range(G):
  >>>     bg = np.random.normal(0, 0.1, np.prod(p)) * (1 - common_features) + common_effects
  >>>     mu = RH(x[2], RH(x[1], RH(x[0], np.reshape(bg, (p[0], p[1], p[2]), "F") )))
  >>>     y[:, :, :, g] = np.random.normal(0, 1, (n)) + mu
  
  ##fit model for range of lambda and zeta
  
  >>> zeta = np.array([0.1, 1, 10, 100])
  >>> fit = softmaximin(y, x, zeta = zeta, penalty = "lasso", alg = "npg")
  >>> yhat = predict(fit, x)
   
  #Array data and wavelets
  ##size of example

  >>> G = 5; 
  >>> p = n = np.array([2**2, 2**3, 2**4])
  
  ##wavelet design

  >>> x = "la8"

  ##common features and effects
  
  >>> common_features = np.random.binomial(1, 0.1, np.prod(p)) #sparsity of common effects
  >>> common_effects = np.random.normal(size = np.prod(p)) * common_features
  
  ##group response

  >>> y = np.zeros((n[0], n[1], n[2], G))
  >>> for g in range(G):
  >>>     bg = np.random.normal(0, 0.1, np.prod(p)) * (1 - common_features) + common_effects
  >>>     mu = iwt(np.reshape(bg, (p[0], p[1], p[2]), "F"))
  >>>     y[:, :, :, g] = np.random.normal(0, 1, (n)) + mu
  
  ##fit model for range of lambda and zeta
  
  >>> zeta = np.array([0.1, 1, 10, 100])
  >>> fit = softmaximin(y, x, zeta = zeta, penalty = "lasso", alg = "npg")
  >>> modelno = 10
  >>> zetano = 2
  >>> yhat = predict(fit, x)
  >>> yhat[zetano][:,:,:, modelno]
  
  #Non-array data
  ##size of example

  >>> G = 10
  >>> n = np.random.choice(np.arange(100,500,1), G) #sample(100:500, G); 
  >>> p = 60
  >>> x = [None] * G
  
  ##group design matrices

  >>> for i in range(len(x)):
  >>> x[i] = np.random.normal(0, 1, (n[i], p))
  
  ##common features and effects

  >>> common_features = np.random.binomial(1, 0.1, np.prod(p)) #sparsity of common effects
  >>> common_effects = np.random.normal(size = np.prod(p)) * common_features
  
  ##group response

  >>> y = [None] * G
  >>> for g in range(G):
  >>>     bg = np.random.normal(0, 0.5, np.prod(p)) * (1 - common_features) + common_effects
  >>>     mu = np.matmul(x[g], bg)
  >>>     y[g] = np.random.normal(0, 1, n[g]) + mu
  
  ##fit model for range of lamb and zeta
  
  >>> fit = softmaximin(y, x, zeta = zeta, penalty = "lasso", alg = "npg")
  >>> yhat = predict(fit, x)

   """    
  
 n = fit["dimobs"]
 p = fit["dimcoef"]
 dim_model = fit["dimmodel"] #nNone f0r general
 nzeta = len(fit["zeta"])
 out = [None] * nzeta

 if(type(x) == list or type(x) == str): ##array model
   if(not len(x) == dim_model):
    return "the length of x must be equal to the dimension of the model!"
   if(type(x) == list): ##custom non wavelet model 
    nx = [None] * len(x)
    px = [None] * len(x)
    for i in range(len(x)):
      nx[i] = x[i].shape[0]
      px[i] = x[i].shape[1]
    if(not px == p):
      return "the parameter dimension of the supplied data is not equal to the parameter dimension of the model "
    if(dim_model == 1):
     x.append(np.ones((1, 1)))
     x.append(np.ones((1, 1)))
    elif(dim_model == 2):
     x.append(np.ones((1, 1)))
   else: #wavelet
     nx = n
     px = p  

   for z in range(nzeta):  
     nlambda = len(fit["lamb"][z])
     res = np.zeros((nx[0], nx[1], nx[2], nlambda), order = "F")
     for m in range(nlambda):
      beta = np.reshape(fit["coef"][z][:, m], (px[0], px[1], px[2]), order = "F")
      if(type(x) == str):
       res[:,:,:, m] = iwt(beta, wf = x)
      else:
       res[:, :, :, m] = RH(x[2], RH(x[1], RH(x[0], beta)))
     out[z] = res
 elif(type(x) == np.array): #varying design general model
   nx = x.shape[0]
   px = x.shape[1]
   if(not px == p):
      return "column dimension of the new data x is not equal to the number of coefficients in the model"
   
   for z in range(nzeta):
    nlambda = len(fit["lamb"][z])
    res = np.zeros((nx, nlambda), order = "F")
    for m in range(nlambda):
     res[:, m] = np.maltmul(x, fit["coef"][: , m])
    out[z] = res
   
  #return("dimension of new data inconsistent with existing data"))
 
 return out
