pysmme package
==============
Efficient procedure for solving the Lasso or SCAD penalized soft maximin problem.

This software implements two proximal
gradient based algorithms (NPG and FISTA) to solve two different forms of the soft
maximin problem from Lund et al., 2022 see [1] https://doi.org/10.1111/sjos.12580:

1) For general group specific design the soft maximin problem is solved using the 
NPG algorithm.

2) For fixed identical design across groups, the estimation procedure uses 
either the FISTA algorithm or the NPG algorithm in the following two cases:
i) For a tensor design matrix the algorithms use array arithmetic  to 
avoid the design matrix and speed computations ii) For a wavelet based design 
matrix the algorithms use the pyramid algorithm to avoid the design matrix and 
speed up computations.

Multi-threading is possible when openMP is available.

pysmme.tools module
-------------------

.. automodule:: pysmme.tools
   :members:
   :undoc-members:
   :show-inheritance:

pysmme.transforms module
------------------------

.. automodule:: pysmme.transforms
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: pysmme
   :members:
   :undoc-members:
   :show-inheritance:
