import numpy as np
import theano.tensor as T 
from theano import function

#EX1
#Define scalars
a = T.dscalar('a')
b = T.dscalar('b')
c = T.dscalar('c')
d = T.dscalar('d')
e = T.dscalar('e')
#Define the output function
# f = ((a - b + c) * d)/e
# g = function([a,b,c,d,e],f)

# print "Theano o/p:",g(1,2,3,4,5)
#EX2
#Define the vectors
v1 = T.dmatrix('v1')
v2 = T.dmatrix('v2')
v3 = T.dmatrix('v3')
v4 = T.dmatrix('v4')

v5 = ((a*v1) + (b-v2) - (c+v3))*v4 

fv = function([v1,v2,v3,v4,a,b,c],v5) 

a_data = np.array([[1,1],[1,1],[1,1],[2,2]])
b_data = np.array([[2,2],[2,2],[1,1],[2,2]])
c_data = np.array([[5,5],[5,5],[1,1],[2,2]])
d_data = np.array([[3,3],[3,3],[1,1],[2,2]])




print "Theano o/p:",fv(a_data,b_data,c_data,d_data,1,2,3)
print v5.type()

