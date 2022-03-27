import matplotlib.pyplot as plt
import numpy as np
from pysmme.tools import softmaximin
from pysmme.tools import predict
from pysmme.transforms import RH, iwt, wt
#array data
##size of example
G = 3; 
n = np.array([65, 26, 13])
p = np.array([13, 5, 4])

##marginal design matrices (Kronecker components)
x = [None] * 3 
for i in range(len(x)):
    x[i] = np.random.normal(0, 1, (n[i], p[i]))
     
##common features and effects
common_features = np.random.binomial(1, 0.1, np.prod(p)) #sparsity of common effects
common_effects = np.random.normal(size = np.prod(p)) * common_features

##group response
y = np.zeros((n[0], n[1], n[2], G))
for g in range(G):
    bg = np.random.normal(0, 0.1, np.prod(p)) * (1 - common_features) + common_effects
    mu = RH(x[2], RH(x[1], RH(x[0], np.reshape(bg, (p[0], p[1], p[2]), "F") )))
    y[:, :, :, g] = np.random.normal(0, 1, (n)) + mu

##fit model for range of lambda and zeta
zeta = np.array([0.1, 1, 10, 100])
#zeta = np.array(10)

fit = softmaximin(y, x, zeta = zeta, penalty = "lasso", alg = "npg")
modelno = 10
zetano = 2
betahat = fit["coef"][zetano][:, modelno]
 
f, ax = plt.subplots(1)
ax.plot(common_effects, "r+")
ax.plot(betahat)
plt.show()

#Array data and wavelets
##size of example
G = 5; 
p = n = np.array([2**2, 2**3, 2**3])
#wavelet design
x = "la8"

##common features and effects
common_features = np.random.binomial(1, 0.1, np.prod(p)) #sparsity of common effects
common_effects = np.random.normal(size = np.prod(p)) * common_features

##group response
y = np.zeros((n[0], n[1], n[2], G))
for g in range(G):
    bg = np.random.normal(0, 0.1, np.prod(p)) * (1 - common_features) + common_effects
    mu = iwt(np.reshape(bg, (p[0], p[1], p[2]), order = "F"))
    y[:, :, :, g] = np.random.normal(0, 1, (n)) + mu

##fit model for range of lambda and zeta
zeta = np.array([0.1, 1, 10, 100])
fit = softmaximin(y, x, zeta = zeta, penalty = "lasso", alg = "npg")
modelno = 5
zetano = 3
betahat = fit["coef"][zetano][:, modelno]
 
f, ax = plt.subplots(1)
ax.plot(common_effects, "r+")
ax.plot(betahat)
plt.show() 

yhat = predict(fit, "la8")
plt.plot(yhat[zetano][:,2,2,modelno])

##Non-array data
##size of example
G = 10
n = np.random.choice(np.arange(100,500,1), G) #sample(100:500, G); 
p = 60
x = [None] * G

 ##group design matrices
 #for(g in 1:G){x[[g]] = matrix(rnorm(n[g] * p), n[g], p)}
for i in range(len(x)):
    x[i] = np.random.normal(0, 1, (n[i], p))

 ##common features and effects
common_features = np.random.binomial(1, 0.1, np.prod(p)) #sparsity of common effects
common_effects = np.random.normal(size = np.prod(p)) * common_features

##group response
y = [None] * G
for g in range(G):
    bg = np.random.normal(0, 0.5, np.prod(p)) * (1 - common_features) + common_effects
    mu = np.matmul(x[g], bg)
    y[g] = np.random.normal(0, 1, n[g]) + mu

 ##fit model for range of lamb and zeta
fit = softmaximin(y, x, zeta = zeta, penalty = "lasso", alg = "npg")
betahat = fit["coef"]

 ##estimated common effects for specific lamb and zeta
modelno = 6 
zetano = 2
f, ax = plt.subplots(1)
ax.plot(common_effects, "r+")
ax.plot(betahat[zetano][:, modelno])
plt.show() 

 
 

""" d=np.reshape(np.arange(1,65,1), (4, 4, 4), order = "F")

d=np.reshape(np.array([1.042495e-02,  2.818002e+00  ,1.042495e-02 , 2.818002e+00  ,1.042495e-02
,   2.818002e+00,  1.042495e-02,  2.818002e+00 , 4.169982e-02,  4.169982e-02
, 1.127201e+01  ,1.127201e+01 , 4.169982e-02  ,4.169982e-02 , 1.127201e+01
,  1.127201e+01 , 1.667993e-01,  1.667993e-01 , 1.667993e-01,  1.667993e-01
,  4.508803e+01 , 4.508803e+01,  4.508803e+01 , 4.508803e+01, -5.645764e-15
, -2.847085e-16 ,-2.921765e-15, -2.356061e-15 ,-1.146342e-15,  3.043347e-16
, -1.611225e-15 ,-5.780490e-16,  6.926968e-16 , 4.996004e-16, -1.800535e-15
,-4.440892e-16 ,-4.265902e-15 ,-8.604228e-16 ,-3.066666e-15 ,-6.383782e-16
,  2.360959e-15, -6.739401e-16, -8.770762e-15, -9.325873e-15 , 9.256484e-15
,  1.354515e-14,  4.440892e-15, -1.199041e-14,  9.578527e-16 , 4.318396e-16
, -4.272894e-16,  8.997692e-16, -3.608994e-15,  1.176928e-16, -2.251106e-15
, -3.993022e-16, -6.945144e+00, -2.778057e+01, -1.111223e+02,  1.703550e-14
,  8.326673e-15, -2.398082e-14, -1.711401e-14 , 2.600000e+02]),(4,4,4), order = "F")

d1=iwt(d)
wt(d1)

 """