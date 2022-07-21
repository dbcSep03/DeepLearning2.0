# python与矩阵
矩阵的逆：np.linalg.inv（matrix）
矩阵对角线：matrix.diagonal（）
矩阵的迹：matrix.diagonal（）.sun（）
矩阵的秩：np.linalg.matrix_rank(matrix ) 
矩阵的行列式：np.linalg.det(matrix)
返回均值：np.mean(matrix) 
返回方差：np.var(matrix)
返回标准差：np.std(matrix)
计算两个向量的点积：np.dot(vector_a, vector_b)/vector_a @ vector_b
```python
## 导入库函数
import numpy as np
import scipy
#计算矩阵的逆
a = np.array([[1,1,-1],[1,2,1],[3,1,-2]])
b = np.linalg.inv(a)
"""
array([[-1. ,  0.2,  0.6],
       [ 1. ,  0.2, -0.4],
       [-1. ,  0.4,  0.2]])
"""
#计算矩阵的迹
c = a.diagonal()
#array([ 1,  2, -2])
#计算矩阵的秩
d = np.linalg.matrix_rank(a)
#3
#计算矩阵的行列式
e = np.linalg.det(a)
#5.000000000000001
#计算平均值
f = np.mean(a)
#0.7777777777777778
#计算方差
g = np.var(a)
#1.9506172839506175
#计算标准差
h = np.std(a)
#1.3966450099973928
#计算点乘
m = np.array([1,2,3])
n = np.array([4,5,6])
x = np.dot(m,n)
y = m@n
print(x)
print(y)
# 32

```
## 稀疏矩阵
创建稀疏矩阵:from scipy import sparse
创建压缩稀疏行（CSR）矩阵matrix_sparse = sparse.csr_matrix(matrix)
```python
from scipy import sparse
d = np.array([[0,0,1,0],[2,0,0,1],[0,3,0,0],[1,2,0,0]])
matrix_sparse = sparse.csr_matrix(d)
matrix_sparse
"""
<4x4 sparse matrix of type '<class 'numpy.intc'>'
	with 6 stored elements in Compressed Sparse Row format>
"""
```
##将字典转换为矩阵
加载库：from sklearn.feature_extraction import DictVectorizer
创建 DictVectorizer 对象dictvectorizer = DictVectorizer(sparse=False)
将字典转换为特征矩阵：features = dictvectorizer.fit_transform(data_dict)
查看特征矩阵的列名：dictvectorizer.get_feature_names()

```python
from sklearn.feature_extraction import DictVectorizer
data_dict=[{"red":1,"Yellow":2},{"blue":3,"Yellow":1},{"blue":5,"red":2}]
dictvectorizer = DictVectorizer(sparse=False)
features = dictvectorizer.fit_transform(data_dict)
print(features)
print(dictvectorizer.get_feature_names())
"""
[[2. 0. 1.]
 [1. 3. 0.]
 [0. 5. 2.]]
['Yellow', 'blue', 'red']
"""
```

