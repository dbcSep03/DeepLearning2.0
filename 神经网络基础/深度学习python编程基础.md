# 深度学习python编程基础
## Numpy的使用
> import Numpy

* numpy.array()：数组的创建，生成多维数组等
* numpy.shape()：获得每个维度的元素的数量
* numpy.size()：获得元素的总个数
* len()：获得初始维度的元素个数
* numpy.zeros(n)：生成n个全为零元素的一维数组
* numpy.ones(n)：生成n个全为一元素的一维数组
* numpy.random.rand(n)：生成n个全为随取数的一维数组
* numpy.ones((n,m))：生成形状为(n,m)的全一数组
* numpy.random.rand((n,m))：生成形状为(n,m)的全随机数的数组
* numpy.arange(起始值，终止值，步长)：不包含终止值，若省略起始值，和步长，默认为起始值为0，步长为1
* numpy.linspace(起始值，终止值，元素数量)：包含终止值，如果元素数量省略，则为50
* numpy.reshape(a,(n,m))，a.reshape(m,n)：将数组a转化为形状为(n,m)的数组，变换前后元素数量要一致。若某一个参数为-1，若只有一个参数，则为一维数组，若多个参数，则自动用元素个数计算出-1对应维度的值

## 数组的运算

* +/-/*:进行数乘运
* 在矩阵的数乘运算中，形状不同，通过`广播机制`机制可以进行运算，即通过对某一维度的拓展，是两个矩阵形状相同，进行计算。
* numpy.dot(a,b):对a（p * q）与b（q * r）进行矩阵运算得到c（p*r）的矩阵

## 数组的访问

对于单个元素的访问：通过[]进行索引，且起始值都为`0`。对于二维数组的访问用a[n][m]或者a[(n,m)]进行访问（几维数组就需要几个索引值）。
当指定的索引值小于数组维度数量，则返回对应的维度值
指定条件的替换：d[条件]
```
d=numpy.array([0,1,2,3,4,5,6,7,8,9])
print(d[d%2==0])
--------------
[0 2 4 6 8]
```
指定数组作为索引：实现对多个元素的集中访问
```
e=numpy.zeros((3,3))
f=numpy.array([8,9])

e[numpy.array([0,2]),numpy.array([0,1])]=f #将（0，0）和（2，1）替换为f
---------------------
[[8. 0. 0.]
 [0. 0. 0.]
 [0. 9. 0.]]
```
## 切片

> 数组名[此索引之后：此索引之前]通过冒号（：）指定对数组的访问。
数组名[此索引之后：此索引之前：步长]
数组名[:] :提取数组的全部元素
多维数组的提取：数组名[,]:在[]中进行每一维度的索引，维度与维度之间用逗号隔开

## 轴与transpose方法
numpy包含轴的概念，轴即为坐标轴。轴的数量与维度数量相等。
二维数组中，纵向轴是axis=0，横向轴为axis=1。
数组名.transpose(0,1):即为横向和纵向调换（利用数组名.T进行转置也可以）
三维数组中，由外向内的竖轴（z轴）是axis=0，平面数轴axis=1，平面横轴axis=2。
## Numpy的函数
### sum函数
numpy.sum():对所有数求和
numpy.sum(数组名,axis=n)：对于axis=n方向的元素求和,并且降维了，返回一个数组
numpy.sum(数组名，axis=n,keepdims=True)：对于axis=n方向的元素求和,并且维度不变，返回一个数组
### max函数
numpy.max():求所有元素的最大值
numpy.max(数组名，axis=n)：求axis=n维度上元素的最大值，返回一个数组
### argmax函数
numpy.argmax(数组名，axis=n)：在axis=n上最大值的索引值
```
a=numpy.array([[0,4],[2,3]])
print(numpy(argmax(a,axis=0)))
--------------------
[1 0]
```
### where函数
```
numpy.where(条件，满足条件的情况下的值，不满足条件的情况下的值)
```
## Matplotlib
### Matplotlib的导入
作用：图标的绘制，图像的显示，简单动画的制作
实现图表的绘制操作，导入Matplotlib中的Pyplot模块
如果需要使用Jupyter Notebook对matplotlib图表以内嵌方式显示，需要在开头加入%mayplotlib inline语句
```
%matplotlib inline

import numpy
import matplotlib.pyplot
```
### 绘制图表
设定x的个数，y与x的关系， matplotlib.pyplot.plot(x,y)， matplotlib.pyplot.show()将图表显示出来.
### 图表的修饰
*  matplotlib.pyplot.xlabel("")：x轴
*  matplotlib.pyplot.ylabel("")：y轴
* matplotlib.pyplot.title(""):抬头
*  matplotlib.pyplot(x,y,label=' ',linestyle="")：label设置图例，linestyle设置为线条形式
### scatter
 matplotlib.pyplot.scatter(x,y,marker='')：marker设置散点的点形状
### 图像的显示
inshow函数，将数组作为图像显示。











