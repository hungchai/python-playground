import numpy as np
import pandas as pd

labels=np.arange(20)
#print(labels[:,np.newaxis])
#print(labels[:,None])
# print(np.arange(8))
# print((np.arange(8)==labels[:,None]).astype(int))

y = np.mgrid[0:5,0:10].swapaxes(0,2).swapaxes(0,1)
print(y);

x = np.array([
[ 0,  1,  2],
[ 3,  4,  5],
[ 6,  7,  8],
[ 9, 10, 11]])

rows = np.array(
[[0,1,3],
[2,2,2],
[3,3,3]], dtype=np.intp)

columns = np.array(
[[0,2,1],
[0,2,2],
[0,1,1]], dtype=np.intp)

#print(x[:,2:3])
#print(x[:,0:3, None])
print(x[rows])    
print(x[rows,columns])      
print(x[rows,columns,None])                  
                  