# python机器学习入门

### 机器学习基本框架

算法分为：有监督学习和无监督学习

框架体系分为：

1. 分类，一般用于二元或类型数据集，不适用于顺序类型或全集子集的分类；
2. 回归，分析一个因变量和多个解释变量之间的关系。其中逻辑回归算法，采用回归形式，用于分类问题；
3. 聚类分析，无监督学习，将样本中某种维度上相似的对象归到一个簇中；
4. 关联规则，发现隐藏在数据中的内在联系，揭示事物特征间联系；

### 机器学习实施流程

1. 收集样本数据，比如仿真数据，实测数据或爬虫数据；
2. 数据预处理，占据整个工作流程80%以上时间，包括分类型和连续型数据转变，缺失数据，异常数据，重复数据的修正；
3. 变量预处理，转换变量格式，适应目标模型的输入格式；
4. 模型训练，选择更加合适的算法，尽可能提高准确性和泛化性；
5. 模型应用，结合实际情况，优化算法，例如在线学习；

### Python机器学习常用库

主要包括科学计算包（Numpy）、数据分析工具（Pandas）、数值计算包（Scipy）、绘图工具包（Matplotlib）、机器学习包（Scikit-learn），涵盖数据导入，整理，数据处理，可视化，数值计算和算法运行等方面。

python扩展资源包下载地址：https://www.lfd.uci.edu/~gohlke/pythonlibs/

上述包的集成管理工具Anaconda，该软件包括python和一系列机器学习的包；

Jupyter Notebook模式，支持机器学习实时呈现，可视化，网页共享功能；

1. ##### numpy,核心是数组，包括对数组的运算和数学运算；

~~~python
import numpy as np
array=np.array([[1,2,3],[4,5,6]]) #创建二维数组，打印出来为两行三列矩阵
print(array)
print('数组维度为：',array.shape)
print('数据类型',array.dtype)
print('元素个数',array.size)
~~~

![image-20201028100947100](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028100947100.png)

~~~python
np.linspace(0,1,5) # 0到1之间均匀取5个数
np.zeros((2,3)) # 两行三列的置0数组
np.random.random(10) # 10个0到1随机数数组
arr=np.random.random((2,3)) # 两行三列0到1随机数组
print(arr*2) # 对数组的简单运算
print(array[0,1:2]) # 对数组索引，第1列内容
np.sort(arr) #由小到大排列
np.sum(arr) #求和
np.mean(arr) # 均值
np.std(arr)  # 标准差
np.dot(arr,arr1) #矩阵相乘
~~~

2. ##### pandas补充（前一部分见数据可视化.md）

~~~python
import pandas as pd
df1=pd.read_csv('D:/hydraulic_cylinder_test/measured_out1.csv')
print(df1.head()) # 查看前5行数据
print(df1.tail()) # 查看后5行数据
print(df1.index) 
print(df1.columns) # 分别为数据集的索引行和索引列
print(df1.describe()) # 显示整个数据的统计性分析结果
~~~

![image-20201028105829896](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028105829896.png)

~~~python
import pandas as pd
df=pd.DataFrame({'a':[1,5,3,4],'b':[4,5,6,7]}) #由字典创建序列，值不够补NaN
df1=df.sort_values(by='a') # 以a列的值升序排列，行序号发生改变
print(df1) 
print(df['b'])  # 索引属性b列的值，带序号输出
print(df.iloc[0:2,1:2]) # 索引具体的序列中的值，0：2索引0和1行，1：2索引第1列，输出带行列标
df['c']=pd.Series([1,2]) # 补充一列，值不够补NaN
print(df)
~~~

![image-20201028144253657](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028144253657.png)

3. ##### Scipy，在Numpy基础上做更加复杂运算，如积分，插值，优化算法，稀疏矩阵；

~~~python
import scipy.stats as stats  # 统计、假设检验
y=stats.norm.rvs(size=20,loc=0,scale=1) 
# y=stats.norm.rvs(0，1，size=20) 和上式均产生均值0，方差1的20个正太分布的随机数
stats.ttest.ind(a,b) # 对a，b两个分布T检验，输出T统计量和其p值
print(y)

from scipy import linalg  # 线性代数操作
import numpy as np
arr=np.array([[1,2],[3,4]])
print(linalg.det(arr)) # 计算行列式
iarr=linalg.inv(arr) # 计算矩阵的逆
print(iarr)

from scipy import optimize  #优化算法

~~~

4. ##### Matplotlib绘图工具库，补充饼图，散点图，柱状图

~~~python
import numpy as np
import matplotlib.pyplot as plt
labels=['A','B','C','D']  # 规定饼状图各部分名称
percent=[10,30,40,10]   # 各部分的比例
explode=[0,0,0.2,0]   # 各部分离中点的距离
plt.pie(x=percent,labels=labels,autopct='%.2f%%',explode=explode,shadow=True)
# x代表比例，labels标签，autopct精度，shadow是否带阴影
plt.show()
~~~

![image-20201028160347703](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028160347703.png)

~~~python
x=np.linspace(0,10,50)
y=x**2+2*x+np.sin(x)
plt.scatter(x,y,color='r',marker='.') # marker规定点形状
bar1=[35,40.5,90,15]
plt.bar(range(4),bar1) # range的数量必须和柱形一样多
plt.show()
~~~

![image-20201028162343961](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028162343961.png)

5. ##### Scikit-learn（机器学习包)，提供一系列监督学习和无监督学习算法，建立在Scipy库之上

~~~python
# 加载自带数据库iris,data包括四种特征，target代表特征下三种花分类
from sklearn import datasets # 导入sklenrn自带数据库
iris=datasets.load_iris()
data=iris.data
target=iris.target
print(data,target)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.3)
# 由train_test_split函数将数据分为训练集和测试集，测试集占30%
print(x_train.shape)  # 输出序列有几行几列

~~~

由Scikit-learn的包对数据三种预处理：归一化，标准化，正则化；均为按列来

~~~python
# 归一化方法(normalization) (x-min(x))/(max(x)-min(x))
from sklearn import preprocessing # 从sklearn导入前处理的包
import numpy as np
data=np.matrix(np.array([[1,2,3,4],[4,5,6,7],[9,3,7,8]])).T 
# 由[]建立的是列表，依次是列表，元组，字典，集合；np.array()将其变成数组，np.matrix()将其变为矩阵，矩阵和pandas.DataFrame形式可用.T做转置；
min_max=preprocessing.MinMaxScaler() # MinMaxScaler为预处理模块中一个类，该步骤首先要通过该类创建一个对象，实体化
x_train=min_max.fit_transform(data) # 调用该函数实现归一化，但是按列归一化，如果想按行归一，先转置
print(x_train.T)

~~~

![image-20201028212400235](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201028212400235.png)

~~~python
# 标准化方法(standardization) (x-mean(x))/std(x)
from sklearn import preprocessing
import numpy as np
x=np.array([[1,2,3],[7,6,3],[9,7,8]])
x_std=preprocessing.scale(x)   # 调用该函数实现标准化，按列来；加axis=1，可调整到行方向
print(x_std)
print(x_std.mean(axis=0))  # axis=0代表沿着列方向，即跨行；axis=1代表沿着行方向，即跨列；
print(x_std.std(axis=0))  # 分别计算对应列的均值，方差
~~~

![image-20201029092549549](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201029092549549.png)

~~~python
# 正则化(regularization) 求样本的p-范数，||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
from sklearn import preprocessing
import numpy as np
x=np.array([[1,2,3],[7,6,3],[9,7,8]])
x_regula=preprocessing.normalize(x, norm='l2',axis=0) #ln的n代表用n阶范数，axis=0代表按列来取范数再除原项；
print(x_regula)
~~~

![image-20201029100513804](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201029100513804.png)

由Scikit-learn的包对iris数据库调用逻辑回归，决策树，K近邻算法和支持向量机判断花朵类型；

~~~python
from sklearn import datasets
iris=datasets.load_iris()
data=iris.data
classes=iris.target
from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(data,classes,test_size=0.3)
# y_train,y_test=model_selection.train_test_split(classes,test_size=0.3)若分别分类则无法将特征和结果完全匹配
# 将特征和对应的结果分为训练集测试集两部分
from sklearn import linear_model
lr=linear_model.LogisticRegression() # 为逻辑回归算法类建立对象
lr.fit(x_train, y_train) # 用逻辑回归算法拟合
print(lr.coef_) # 打印逻辑回归求解参数

~~~

![image-20201029164628187](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201029164628187.png)

~~~python
import mean_aquare
y_pred=lr.predict(x_test)
print(y_pred)
print(y_test)
from sklearn import metrics
print(metrics.mean_squared_error(y_test,y_pred))
#print(mean_square.square(y_test,y_pred))
print(type(y_test),type(y_pred))

#在另一个文件mean_aquare.py中定义square函数
import math
import numpy as np
def square(x,y):
    z=len(x)
    sum1=sum((x-y)**2)  # x**2对x（numpy数组）中每一个元素平方，若对列表无法进行该操作
    square1=math.sqrt(sum1/z)
    return square1
#if __name__=="__main__":  # 定义程序入口，只有当程序名字为自己时运行，即import调用时不运行，下图中自动运行了该文件程序
    m=np.array([1,2,3,4])
    n=np.array([1,2,3,7])
    print(square(m,n))
~~~

![image-20201030181713565](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201030181713565.png)

运用支持向量机算法测试

~~~python
from sklearn import svm
from sklearn import matrics
svm=svm.SVC()             
svm.fit(x_train,y_train)
y_pred_svm=svm.predict(x_test) # 先将类实体化，用训练集拟合，然后预测
print(matrics.mean_squared_error(y_test,y_pred_svm))
#输出0.111111 比上面用逻辑回归算法均方差小
~~~



### 机器学习拟合预测结果准确性判断

##### 评估方法：

对训练好模型的泛化误差评估，需要选取测试集，由测试集上”测试误差“作为泛化误差的近似，选取测试集方法如下：

1. 留出法，对测试集选取时分层采样，尽量在数据集各个类别中都选到；
2. 交叉验证法，先将数据集划分为k（通常为10）个大小相似的互斥子集（分层采样划分），每次采用k-1个子集作为训练集，剩下的作为测试集，用k次测试结果的均值作为评估结果；
3. 自助法，从m个样本的数据中随机采样m次，构成训练集，没采到的作为测试集。主要用在数据集较小，难以有效划分训练集使用，但可能会引入估计偏差；

##### 模型调参：

学习算法中参数的设定对于最终模型训练结果有很大影响；参数较少时，可以设定步长调节参数，依次训练模型判断模型性能；

在训练集中分出一部分调参，为验证集，基于验证集上的性能进行模型选择和调参，测试集为模型在实际使用中的泛化能力；

##### 性能度量：

1. 均方误差（mean square error），常用于线性回归任务；
2. 错误率与精度，判断错误占总样本比例，判断正确占总样本比例；
3. 查准率，查全率，F1。TP(真正例)，FP(假正例)，FN(假反例)，TN(真反例)，TP+FP+FN+TN=样例总数，P(查准率)=TP/(TP+FP),判断为故障的情况下，真正是故障的比例； R(查全率)=TP/(TP+FN)，在真正有故障的情况下，判断出有故障的比例；

F1度量，基于查准率和查全率的调和平均，Fβ为加权调和平均，调和平均更重视较小值；
$$
F_1=\frac{2\times P\times R}{(P+R)}  \quad\quad
F_\beta=\frac{(1+β^2)\times P\times R}{(β^2\times P)+R}
$$
β=1为标准的F1形式，β>1，查全率有更大影响，β<1，查准率有更大影响；

4. ROC和AUG，很多学习器产生预测值后，与分类阈值比较，大于阈值为正类，否则为反类；

   ROC(receiver operating characteristic)，由学习器的预测结果对样例排序，逐个把样本当作正例预测，截断点在前，更重视查准率，在后，更重视查全率。求出相应的TPR(真正例率)和FPR(假正例率)，作为纵轴，横轴，即为ROC图，其与坐标轴围成的面积即为AUG,面积越大，则模型的训练效果越好。
   $$
   TPR=\frac{TP}{TP+FN}  \quad \quad FPR=\frac{FP}{TN+FP}
   $$

5. 代价敏感错误率

   不同类型的错误代价不同，可赋予错误”非均等代价“，cost_ij代表将第i类样本预测为第j类的代价，一般情况预测正确无损失，cost_ii为0；
   $$
   E_(f;D;cost)=\frac{1}{m}(\sum_{x\in D^+}{f(x_i\ne y_i)\times cost_01}+\sum_{x \in D^-}{f(x_i\ne y_i)\times cost_10})
   $$
   计算除代价敏感错误率后，作为分类任务的代价敏感性度量；

   

# 机器学习基本算法

### 1. 线性回归算法

**拟合**：通过数据之间的关联建立一种近似的函数关系，可以不经过任何点，却能描述这些数据基本规律的曲线；基于”偏差的平方和最小“，形成了最小二乘法；

**回归**：相对于拟合增加了预测理念，任何测量的现象都有平均值，回归就是不断向平均值逼近；

求解单变量线性回归算法中参数θ_0 ，θ_1,常规方法最小二乘，为此引入成本函数(cost function)
$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}{(h_\theta(x^i)-y^i)^2}
$$
将最小二乘法问题转化为求解成本函数最小问题，常规方法求成本函数偏导，令其为0；
$$
h_\theta(x)=\theta_0+\theta_1 x
$$

$$
\frac{\partial}{\partial \theta_0}J(\theta)=\frac{1}{m}\sum_{i=1}^{m}{(\theta_0+\theta x^i-y^i)\times 1=0}
$$

$$
\frac{\partial}{\partial \theta_1}J(\theta)=\frac{1}{m}\sum_{i=1}^{m}{(\theta_0+\theta x^i-y^i)\times x^i=0}
$$

对偏导函数求解：
$$
\theta_1=\frac{m\sum x^iy^i-\sum x^i\sum y^i}{m\sum (x^i)^2-(\sum x^i)^2}
\quad \quad
\theta_0=\frac{\sum y^i}{m}-\theta_1\frac{\sum x^i}{m}
$$


除均方差，平均绝对值差外，拟合优度(R^2)是判断回归模型拟合程度好坏的常用指标，表示**因变量的y的总变差中可以由回归方程解释的比例**；
$$
R^2=\frac{SSR}{SST}=\frac{\sum (yf^i-\bar{y^i})^2}{(y^i-\bar{y^i})^2}
$$


#### 由线性回归算法对波士顿房屋价格拟合和预测：

首先导入自带数据集boston，选择房间数目’RM‘作为自变量，房屋整体价格作为因变量，对其线性拟合；

~~~python
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn import datasets # sklearn,datasets均为文件夹名称，由所要使用的类或函数，import至少到函数所在.py文件的上一级目录;datasets文件夹下_init.py文件包含load_boston函数
boston=datasets.load_boston() # 导入的数据是字典模式
print(boston.feature_names)  # 数据data属性下的特征名称
print(boston.keys())  # 字典的属性
~~~

![image-20201102160843326](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201102160843326.png)



~~~python
#bos=pd.DataFrame(boston.data)
bos=boston.data
print(bos)
#print(bos[5].head())
#bos_target=pd.DataFrame(boston.target)
bos_target=boston.target  # target为房屋价格排序
#print(bos_target.head())

import matplotlib.font_manager as fm # 导入绘图中的字体选择模块
print(len(bos),len(bos_target))
x=bos[:,5] #导出bos数组中第6列的全部数据
y=bos_target
print(type(x),len(x)) # 输出 <class 'numpy.ndarray'> 506
print(type(y),len(y))
myfont=fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
# 由C盘中带的fonts,选择合适的字体
plt.scatter(x,y)
plt.xlabel('住宅平均房间数',fontproperties=myfont)
plt.ylabel('房屋价格',fontproperties=myfont)
plt.title('RM和MEDV的关系',fontproperties=myfont)
plt.show()
~~~

![image-20201102174430251](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201102174430251.png)

~~~python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
print(type(x_train),x_train.shape, type(y_train),y_train.shape)
# 将数据集划分为训练集和测试集，输出类型为：
#<class 'numpy.ndarray'> (379,) <class 'numpy.ndarray'> (379,)，代表1X0的数组

#from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearRegression #两种调用方法均可用，linear_model为对应文件夹，_base.py为函数所在的python文件，可不写也能找到函数
lr=LinearRegression()
x_train=x_train.reshape(-1,1) #将x_train化为n（不确定）行1列的数组
y_train=y_train.reshape(-1,1)
print(type(x_train),x_train.shape, type(y_train),y_train.shape)
# 输出：<class 'numpy.ndarray'> (379, 1) <class 'numpy.ndarray'> (379, 1)
print(x_train)
lr.fit(x_train, y_train) #类中的fit函数只接收[[],[]]类型的数据，shape为(m,n)，训练模型
print(type(lr.coef_),lr.coef_.shape,lr.coef_) #.coef为求解模型的系数
m=np.array([1,2,3])
print(m,m.shape)   # 由numpy.array([])创建的一维数组
~~~

![image-20201102180409247](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201102180409247.png)

LinearRegression() 类中对初始参数默认设置 

![image-20201102182315165](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201102182315165.png)

fit_intercept=True  表示对输入的数据中心化，normalize=False 表示不对输入的数据标准化处理；

copy_X=true 表示对数据进行复制后再处理，否则新数据将覆盖原数据；

n_jobs=-1，代表使用所有的CPU



对训练好的模型进行评价，包括数据可视化和评价指标：拟合优度，均方差等；

~~~python
x_test=x_test.reshape(-1,1)
y_hat=lr.predict(x_test)   # 预测时的数据也要对其更改成(n,1)的格式
print(y_hat[0:9])

plt.figure(figsize=(10,6)) # 实际值和预测值绘图比较
t=np.arange(len(x_test))
plt.plot(t,y_test,color='r',linewidth=2,label='y_test')
plt.plot(t,y_hat,color='b',linewidth=2,label='y_pred')
plt.legend()
plt.show()
~~~

![image-20201102185505215](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201102185505215.png)

~~~python
from sklearn import metrics
print(lr.score(x_test,y_test))  # 直接由模型计算拟合优度R^2
print(metrics.r2_score(y_test,y_hat))  # 由预测出的数据和实际数据比较得到拟合优度
print("MSE",metrics.mean_squared_error(y_test,y_hat)) #计算均方差
~~~

![image-20201102191329276](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201102191329276.png)



#### 由前面拟合方法中最小化均方差，对成本函数求偏导，偏导数为0来拟合：

~~~python
import math
def linefit(x,y):
    n=len(x)
    sx,sy,sxy,sxx=0,0,0,0
    m=np.arange(n)   # 由numpy.arange(n)函数建立从0到n-1的n项等差数列；
    print(m)
    for i in m:
        sx+=x[i,0]
        sy+=y[i,0]
        sxy+=x[i,0]*y[i,0]
        sxx+=x[i,0]*x[i,0]
    a=(n*sxy-sx*sy)/(n*sxx-sx**2)  # 该公式为偏导数为0时解出的a，b
    b=sy/n-a*sx/n
    return a,b
a1,b1=linefit(x_train,y_train)
print(a1,b1)
y_pred=a1*x_test+b1
from sklearn import metrics
print(metrics.r2_score(y_test,y_pred))  # 由预测出的数据和实际数据比较得到拟合优度
print("MSE",metrics.mean_squared_error(y_test,y_pred)) #计算均方差     
~~~

![image-20201103090903911](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201103090903911.png)



#### 总结机器学习的整体流程

1. 数据准备和需求分析，找到特征变量和目标变量；
2. 数据预处理，首先是数据格式转换，可视化；然后是数据清理，缺失值处理，归一化，标准化等
3. 数据集划分，划分训练集，验证集，测试集，采用不同划分方式以及是否分出验证集；
4. 算法训练；
5. 算法测试，由训练好的算法对测试集数据预测；
6. 算法的评估，通过预测值和真实值的比较，得到模型评价指标的值，比如拟合优度，均方差等，由评价指标调整参数或换另一种算法，不断尝试达到最优；



### 2.线性回归算法进阶

多变量线性回归算法的线性求解

基本模型：
$$
h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n
$$

$$
h_\theta(x)=\sum_{j=0}^{n}{\theta_jx_j}=\theta^Tx
$$

成本函数：
$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}{(h_\theta(x^{i})-y^i)^2}
$$
对各个θ求偏导，并使偏导数为0
$$
\frac{\alpha}{\alpha\theta_k}J(\theta)=\frac{1}{m}\sum_{i=1}^{m}{((\sum_{j=0}^{n}\theta_jx_j^i)-y^i)x_k^i=0}
$$
解得：
$$
\theta=(X^TX)^{-1}X^TY
$$
式子中不可逆的可能原因：1. 自变量间存在高度多重共线性；2. 特征复杂度较高而训练数据较少，特征的数量大于训练数据的数量；



##### 影厅观影人数拟合：（多变量模型拟合）

多变量数据可视化：（数据点的分布情况）

~~~python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.read_csv('D:/test/data/3_film.csv')
print(df.head())
df.hist(xlabelsize=8,ylabelsize=8,figsize=(12,7)) #绘制直方图
df.plot(kind='density',subplots=True,layout=(2,2),sharex=False,fontsize=8,figsize=(12,7))  # 绘制密度图
df.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False,fontsize=8,figsize=(12,7)) # 绘制箱线图，主要查看数据间分布离散程度，差异值，分布差异；
plt.show() #展示了数据的分布情况，横轴为数据值，纵轴为数据的个数，查看数据的分布
~~~

![image-20201106110048716](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201106110048716.png)

![image-20201106110104824](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201106110104824.png)

![image-20201106111556042](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201106111556042.png)

（数据点变量之间的相关程度）

~~~python
names=['num','size','ratio','quality']
correlations=df.corr() # 计算相关系数矩阵
fig=plt.figure()  # 建立一张图像页面
ax=fig.add_subplot(111) # 图像页面上建立n*n个子坐标系，最后的1或2代表靠左还是靠右
#bx=fig.add_subplot(222)
cax=ax.matshow(correlations,vmin=0.3,vmax=1) # 由相关系数矩阵建立热力图
fig.colorbar(cax) # 添加热力图颜色的标记栏
print(correlations)
ticks=np.arange(0,4,1)  # 横轴纵轴标签的数量
ax.set_xticks(ticks)
ax.set_yticks(ticks) 
ax.set_xticklabels(names) # 将名称写在标签上
ax.set_yticklabels(names)
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,10),color='b')
# 绘制散点图矩阵，依次画出各个变量关系的散点图
plt.show()
~~~

![image-20201106191617466](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201106191617466.png)  ![image-20201106191704655](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201106191704655.png)

![image-20201106192948713](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201106192948713.png)

影厅观影人数拟合（最小二乘拟合，用的是解析解）

~~~python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.read_csv('D:/test/data/3_film.csv')
x=df.iloc[:,1:4] # 1:4代表第2，3，4列，0代表第一列
y=df.filmnum # 由列名直接索引
from sklearn.model_selection import train_test_split
x=np.array(x.values)
y=np.array(y.values)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.coef_)  # 打印拟合的参数值 [ 0.37048549 -0.03831678  0.23046921]
y_hat=lr.predict(x_test)
plt.figure(figsize=(10,6))
t=np.arange(len(x_test))
plt.plot(t,y_test,color='r',linewidth=1,label='y_test')
plt.plot(t,y_hat,color='b',linewidth=1,label='y_test')
plt.legend()
plt.show()
~~~

![image-20201107094520029](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201107094520029.png)



影厅观影人数拟合（梯度下降拟合，用数值解）

梯度向量的几何意义是函数变化增加最快的方向，因此沿着梯度的方向更容易找到函数的最大最小值；梯度迭代用于线性拟合，就是在成本函数基础上，用梯度迭代求局部最优解；

学习率：也叫步长，决定了梯度迭代过程中，每一步沿梯度方向前进的长度；学习率小，收敛慢，学习率过大，在最低点处反复震荡；

~~~python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.read_csv('D:/test/data/3_film.csv')
df.insert(1,'ones',1) # 插入第二列，列名ones，值均为1
# 截距项就是θ0的值，为了计算其值，需要加上该列来与之相乘
cols=df.shape[1] # 0为函数，1为列数
x=df.iloc[:,1:cols]
x=np.array(x.values)
y=df.filmnum
y=np.array(y.values)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
def computeCost(x,y,theta):
    inner=np.power(((x*theta.T)-y),2)
    return np.sum(inner)/(2*len(x))

def gradientDescent(x,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    print(temp,parameters,cost)
    for i in range(iters):
        error=(x*theta.T)-y
        for j in range(parameters):
            term=np.multiply(error,x[:,j])
            temp[0,j]=theta[0,j]-((alpha-len(x))*np.sum(term))
        theta=temp
        cost[i]=computeCost(x,y,theta)
    return theta,cost

alpha=0.000001
iters=100
theta=np.matrix(np.array([0,0,0,0]))
g,cost=gradientDescent(x,y,theta,alpha,iters)
print(g)
~~~























### 3. 逻辑回归算法

logistic regression 在线性回归算法的基础上，将y的数值划分多类，实现对事务的分类拟合；

它使用逻辑函数，通常是Sigmiod函数将输出变量的值域压缩到(0,1)，然后设定阈值，通过预测值和阈值比较进行分类；
$$
sigmoid函数：g(z)=\frac{1}{1+e^{-z}}
$$
![image-20201103104123775](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201103104123775.png)
$$
令z=f(x)=\theta_0+\theta_1 x_1+....+\theta_nx_n
$$
先拟合出线性回归方程f(x)，得到参数值后计算测试集中的预测值，由sigmoid函数转换到(0,1)上，最后和阈值比较分类；

#### 梯度下降法求参数

由于逻辑回归输出值具有不连续的特点，一般用对数似然成本函数最小化对参数求解，很难得到解析解，因此采用梯度下降的迭代法求解；

概率：参数已知时随机变量的输出结果；

似然：在确定的结果下推测产生这个结果的可能参数；例如在机器学习中，利用训练集找出产生这种结果最可能的条件，从而根据这个条件推测未知事件；

似然函数就是将所有产生的结果的概率相乘，为了避免值过小，因此取对数,

例如下式0-1分布的抛硬币，正6次，反4次，计算似然函数,取对数后令其最大；
$$
L(\theta)=L(x_1,x_2,...,x_n;\theta)=\prod_{i=1}^{n}{p(x_i;\theta)}
\quad \quad
L(\theta|x)=p^6*(1-p)^4
$$



### 4.人工神经网络

artificial neural network（ANN）最大特点在于**能够拟合及其复杂的非线性函数**；三层神经网络足以解决任意复杂的分类问题；

神经网络对于信息的处理由许多功能单一的神经元集成，每个神经元负责处理它接受到的简单信息，许多神经元一起处理大量信息，从而让信息处理有并行处理能力；

神经元模型：多输入单输出的非线性阈值器件；单个神经元对信息的处理函数叫做激活函数；

输出矢量：P输入向量     W输入向量依次权重      b偏差信号输入
$$
A=f(W\cdot P+b)
$$
![image-20201111152517105](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201111152517105.png)

对于各个节点的权值，阈值，先赋予一个初始值，然后通过误差的偏差程度，不断迭代调整，得到最优的解决问题的参数；

人工神经网络按照神经元连接方式不同，可分为前馈型，反馈型和自组织型三种；

1. 前馈型，各层处理的信息只向前传送，不能反向相互反馈；
2. 反馈型，每个神经元既可以从外界接受输入，同时也能向外界输出；
3. 自组织，采用无监督学习算法，寻找样本中内在规律，以自组织方式来改变网络参数结构；



##### BP(back propagation)神经网络算法

BP神经网络是多层前馈神经网络，进入输出层的信息反馈回来，对各层的权值阈值修正，因此训练路径包含两个阶段：信息正向传递，误差反向传递；

假设输入层节点数为n，隐含层节点数为r，输出层节点数为m；f1为隐含层节点的激活函数，f2为输出层节点的激活函数；

信号的前向传播：

隐含层第k个节点的输出：
$$
z_k=f_1(\sum_{i=1}^{n}{w_{ik}x_i+\theta_k}) \quad k=1,2....,r
$$
输出层第j个节点的输出：
$$
y_j=f_2(\sum_{k=1}^{n}{v_{kj}z_k+v_j}) \quad j=1,2....,m
$$
误差的方向传播：

样本的网络期望输出 y_i，真实输出 t_j 的误差函数E为：
$$
E=\frac{1}{2}\sum_{j=1}^{m}{(t_j-y_i)^2}
$$
根据误差梯度下降法，计算误差函数对输出层各神经元的偏导数，从而修正权值，阈值；

计算输出层权值调整公式：
$$
\Delta v_{kj}=-\alpha\frac{\partial E}{\partial v_{ij}}=\alpha (t_j-y_i)f_2^\prime z_k
$$
计算隐含层权值调整函数：
$$
\Delta w_{ik}=-\beta\frac{\partial E}{\partial w_{ik}}=\beta[\sum_{j=1}^m{t_i-y_j}f_2^\prime v_{kj}]f_1^\prime x_i
$$
BP神经网络对所选激活函数要求处处可微，因此可采用S型激活函数；根据修改后的权值阈值，与原始的权值阈值相加，计算新值下的误差函数，再根据误差函数进一步对权值阈值修改；

由sklearn.neural_network中MLPClassifier()实现BP算法，但该方式不适合大规模运算，scikit_learn不支持GPU；

神经网络最大特点在于解决了非线性的分类问题，其间原先模型需要的几个参数拓展到几十个，通过对这几十个参数的微调和修正，实现了对人脑学习的模仿；

参数过多导致神经网络的训练时间较长，且不一定能带来最优解；对于非线性问题，SVM（支持向量机）无需调参，高效，逐渐对其代替，因此在人工神经网络基础上的深度学习带来了新的突破；

深度学习网络：

应用最广泛的卷积神经网络(Convolutional neural networks)，相邻层之间的神经单元不是全连接，而是部分连接，使得训练的参数大大减少；同时在隐含层中加入了由卷积层和子采样层构成的特征提取器，大大加强了算法特征学习能力；

深度学习主要特点：

1. 局部感受视野；
2. 权值共享；
3. 降采样；

~~~python
# 人工神经网络的python实现

~~~



























# 降维技术和关联规则挖掘

### 降维处理

原因：1. 数据维度很大时，数据处理所需的时间和空间复杂度都呈指数形式上升；2. 高维情况下，通常会出现空间稀疏情况，样本数据点之间距离的度量随着维度增加反而减弱，由此可能对数据的分离或聚类带来困难；

​      因此要对数据降维，同时保证降维后其中包含的主要信息时相似的；

降维的好处：

1. 减少数据存储空间和计算时间；
2. 减少数据间冗余，提高计算效率；
3. 去除噪声，提高模型性能；
4. 将数据维度减少到二维或三维可进行可视化；
5. 从数据中提取特征可看清数据的分布，提高数据的可理解性；

##### 主成分分析(PCA)基本原理：（主要用于无监督学习）

通过线性组合的方法将多个特征综合为少数特征，且综合后的特征相互独立，又能表示原始特征的大部分信息；

线性组合后第一主成分应选方差最大的，方差越大，表明包含的信息越多；

**PCA步骤：**

1. 原始数据预处理，对数据标准化处理，消除量纲(大值的影响)，实质就是将坐标原点移到样本点中心；
   $$
   x_{ij}^*=\frac{x_{ij}-\bar{x_j}}{\sqrt{var(x_j)}}
   $$
   
2. 计算数据集协方差矩阵，例如5维特征将建立一个5行5列的对称矩阵，第i行j列的元素计算公式：
   $$
   c_{ij}=E{[X_i-E(X_i)][X_j-E(X_j)]}
   $$
   协方差反应数据的相关程度，若是独立变量则协方差为0；

3. 计算协方差矩阵的特征值和特征向量；

4. 选取方差贡献率高的主成分。贡献率就是某个特征值占全部特征值的比重，也就是该成分的方差占全部方差的比重；
   $$
   \varphi_i=\frac{\lambda_i}{\sum_{i=1}^{p}{\lambda_i}}
   $$
   在方差的累计贡献率达到80%，保证综合变量包含原始变量绝大多数信息；选择该特征值对应的特征向量，依次乘特征变量，构成主成分；

~~~python
import matplotlib.pyplot as plt 
import numpy as np 
x=[0.69,-1.31,0.39,0.09,1.29,0.49,0.19,-0.81,-0.31,-0.71]
y=[0.49,-1.21,0.99,0.29,1.09,0.79,-0.31,-0.81,-0.31,-1.01]
z=np.c_[x,y]   #两个list类型的合成一个np.ndarray类型，10行2列
print(z,type(z),type(x))
from sklearn.decomposition import PCA
pca=PCA(n_components=1)  # 要保留的主成分的个数
pca.fit(z)
print('te_zhen_zhi',pca.explained_variance_) # 保留的特征值
print('gong_xian_lv',pca.explained_variance_ratio_) # 所占的贡献率
z_new=pca.transform(z)  #由主成分分析结果将原数组转化为主成分数组
print(z_new)
~~~

![image-20201105101257773](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201105101257773.png)



##### 线性判别分析(LDA)基本原理：（用于有监督学习分类）

寻找一个低维空间投影，使得样本集中的各个类别在低维空间形成最大的类间分散程度和最小的类内分散程度，便于根据特征分类；

先设置高维向低维转换的转换矩阵w，然后根据类内方差最小，类间方差最大原则，由Lagrange乘子法求偏导，求特征值，然后对应的特征向量就是转换矩阵；

特点和局限性：

LDA和PCA共同点：在降维时均采用了矩阵特征分解的思想，空间中表现为投影矩阵的线性映射；

区别：

1. PCA对特征进行了重构，能够最大化样本方差，但无法保留样本的标签信息，所选择的投影具有最大方差的方向，分类不一定最好；
2. LDA算法充分利用了样本类别信息，缺陷在于该方法距离大的类间距忽略距离小的，导致在原始空间中距离较近的样本点在空间发生严重重叠；对其改进方法有1.基于几何平均值的空间选择法(GMSS)，最大最小距离分析法(MMDA)；

~~~python
import matplotlib.pyplot as plt 
import numpy as np 
x=[0.69,-1.31,0.39,0.09,1.29,0.49,0.19,-0.81,-0.31,-0.71]
y=[0.49,-1.21,0.99,0.29,1.09,0.79,-0.31,-0.81,-0.31,-1.01]
z=[0,0,0,0,0,1,1,1,1,1]
#z=np.array(z)
#z=z.reshape(-1,1)
h=np.c_[x,y]
print(h,z)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ida=LinearDiscriminantAnalysis(n_components=1)
ida.fit(h,z)
h_new=ida.transform(h)
print(h_new)
print('quan_zhong',ida.coef_)
print('liebie_junzhi',ida.means_)
~~~

![image-20201105143740701](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201105143740701.png)



### 关联数据挖掘规则

目标：找到所要研究的事务数据库中所有的强关联规则；

概念：项，数据库中最小不可分割的单位；事务，数据库中所有项目集合的子集，事务中每个元素是一个项，一个事务是一个项集；支持度：a%的事务中同时包含X和Y，则X到Y关联规则的支持度为a%；置信度：包含X的项集中b%也包含项集Y，则X到Y的置信度为b%，表明了X的事务中，出现Y的条件概率；频繁项目集：项集X的支持度大于等于给定的最小支持度阈值；强关联规则：X到Y的支持度和置信度分别大于阈值；

步骤：1. 找出频繁项目集； 2. 计算强关联规则；



# 机器学习项目全流程

数据和特征决定了机器学习的上限，而模型和算法只是逼近了这个上限；

案例：房价预测，训练样本量2000个，测试集样本1000个，特征13个，以预测房价的均方根误差为评价指标；

~~~python
#导入数据并对数据预处理；
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
df=pd.read_csv('D:/test/data/13_house_train.csv')
#print(type(df))  导入的.csv数据集类型为pandas.core.frame.DataFrame
#print(df.head())
#print(df.info())
#print(df.describe())
#缺失值处理
print(df[df.isnull().values==True])
df=df.fillna(df.mean()) #df.fillna(method='pad')
#print(df.loc[95])

#数据转换，将文本或日期数据转化为数值型
df['built_date']=pd.to_datetime(df['built_date'])
print(df.head())
# 将日期数据转换成建筑年龄
import datetime as dt 
now_year=dt.datetime.today().year
age=now_year-df.built_date.dt.year
#print(age)
df.pop('built_date')  #删除原先的时间列
df.insert(2,'age',age)
print(df.head())

#文本转化为数值型，对floor列的类型用数字替代
print(df['floor'].unique())  # 提取floor的取值
df.loc[df['floor']=='Low','floor']=0
df.loc[df['floor']=='Medium','floor']=1
df.loc[df['floor']=='High','floor']=2
print(df.head().floor)


# 变量关联性分析
corr_matrix=df.corr()
print(corr_matrix['price'].sort_values(ascending=False))
plt.figure(figsize=(8,3))
plt.subplot(121)
plt.scatter(df['price'],df['area'])
plt.subplot(122)
plt.scatter(df['price'],df['age'])
plt.show()

~~~

原始数据表格：

![image-20201109190622174](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201109190622174.png)

和价格变量协方差大小的排序：

![image-20201109190726259](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201109190726259.png)

面积和寿命对于房屋价格的散点图：

![image-20201109190515774](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201109190515774.png)

~~~python
# 训练模型调参评测；
mlist=[]
# 岭回归算法预测
from sklearn import linear_model
ridge=linear_model.Ridge(alpha=0.1)
ridge.fit(x_train,y_train)
y_hat_ling=ridge.predict(x_test)
from sklearn import metrics
a1=np.sqrt(metrics.mean_squared_error(y_test,y_hat_ling))
mlist.append(a1)

# Lasso回归算法
from sklearn import linear_model
lasso=linear_model.Lasso(alpha=0.1)
lasso.fit(x_train,y_train)
y_hat_lasso=lasso.predict(x_test)
a2=np.sqrt(metrics.mean_squared_error(y_test,y_hat_lasso))
mlist.append(a2)

# 支持向量机回归算法
from sklearn.svm import SVR  #SVC为分类，SVR为回归，分别对应不同的核函数
linear_svr=SVR(kernel='linear')
linear_svr.fit(x_train,y_train)
y_hat_svr=linear_svr.predict(x_test)
a3=np.sqrt(metrics.mean_squared_error(y_test,y_hat_svr))
mlist.append(a3)

# 随机森林回归算法
from sklearn.ensemble import RandomForestRegressor # RandomForestRegressor用于回归，RandomForestClafication用于分类
rf=RandomForestRegressor(random_state=200,max_features=0.3)
rf.fit(x_train,y_train)
y_hat_rf=rf.predict(x_test)
a4=np.sqrt(metrics.mean_squared_error(y_test,y_hat_rf))
mlist.append(a4)
mlist=pd.DataFrame(mlist)
print(mlist,type(mlist))
print(mlist[0].sort_values(ascending=False)) # 对dataFrame类型的数据排序，要选择排序的列名

# 参数调优
# 对支持向量机的回归参数调优
alpha_svr=np.linspace(0.1,1.2,20)
rmse_svr=[]
for c in alpha_svr:
    model=SVR(kernel='linear',C=c)
    model.fit(x_train,y_train)
    y_hat=model.predict(x_test)
    rmse_svr.append(np.sqrt(metrics.mean_squared_error(y_test,y_hat)))
plt.plot(alpha_svr,rmse_svr)
plt.xlabel('alpha')
plt.ylabel('rmse')
plt.show() #由图像可看出，随着SVR中C参数的增大，均方根误差逐渐减小；
~~~

![image-20201109203052328](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201109203052328.png)

![image-20201109202931139](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20201109202931139.png)

