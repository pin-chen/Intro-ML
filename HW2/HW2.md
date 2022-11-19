# HW2

## **Part. 1, Coding (60%)**:

#### 1. (5%) Compute the mean vectors mi (i=1, 2) of each 2 classes on training data

![image-20221031055424790](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20221031055424790.png)

####  2. (5%) Compute the within-class scatter matrix $S_W$ on **training data**

![image-20221031055454664](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20221031055454664.png)

#### 3. (5%) Compute the between-class scatter matrix $S_B$ on **training data**

![image-20221031055524030](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20221031055524030.png)

#### 4. (5%) Compute the Fisher’s linear discriminant $w$ on **training data**

![image-20221031055544114](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20221031055544114.png)

#### 5. (20%) Project the **testing data** by Fisher’s linear discriminant to get the class prediction by K-Nearest-Neighbor rule and report the accuracy score on **testing data** with K values from 1 to 5 (you should get accuracy over **0.88**)

![image-20221031055603258](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20221031055603258.png)

#### 6. (20%) Plot the **1)** **best** **projection line** on the **training data** and show the slope and intercept on the title *(you can choose any value of* **intercept** *for better visualization)* **2)** **colorize the data** with each class **3)** project all data points on your projection line. Your result should look like the below image (This image is for reference, not the answer) 

![image-20221031055629475](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20221031055629475.png)

## **Part. 2, Questions (40%):**

#### (10%) 1. What's the difference between the Principle Component Analysis and Fisher’s Linear Discriminant? 

Principle Component Analysis 是非監督的降維方法，可以降低任意維度，選擇樣本點具有最大 variance 的方向進行降維。

Fisher’s Linear Discriminant 是有監督的降維方法，同時可以拿來做分類，但只能將數據降低一個維度，降維方向選擇不同類的 mean 離最遠且每類的 variance 最小。

#### (10%) 2. Please explain in detail how to extend the 2-class FLD into multi-class FLD (the number of classes is greater than two).

根據 2-class FLD 的 $S_B, S_W$ 的形式定義當有 $k$ 個 class 時的 $S_B, S_W$ 如下：

The within-class covariance matrix when 𝐾 ≥ 2:

$S_W=\Sigma_{k=1}^{K}S_k$, where $S_k=\Sigma_{n\in C_k}(x_n-m_k)(x_n-m_k)^T$ and $m_k=\frac{1}{N_k}x_n$

The extended between-class covariance matrix for 𝐾 > 2:

$S_B=\Sigma_{k=1}^{K}N_k(m_k-m)(m_k-m)^T$, where $m=\frac{1}{N}\Sigma_{n=1}^{N}x_n$

一樣可以得出這個式子 $J(w)=\frac{w^TS_Bw}{w^TS_ww}$

然後對此最佳化求 Maximize $S_B$, Minimize $S_w$ 即為 multi-class FLD

#### (6%) 3. By making use of Eq (1) ~ Eq (5), show that the Fisher criterion Eq (6) can be written in the form Eq (7).

![image-20221031063001125](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20221031063001125.png)

$S_{B}=(m_{2}-m_{1})(m_{2}-m_{1})^{T}$

$S_{W}=\Sigma_{n\in C_{1}}(x_{n}-m_{1})(x_{n}-m_{1})^{T}+\Sigma_{n\in C_{2}}(x_{n}-m_{2})(x_{n}-m_{2})^{T}$

$J(w)=\frac{(m_{2}-m_{1})^{2}}{s_{1}^{2}+s_{2}^{2}}$	Eq(6)

​		   $=\frac{(w^{T}(m_{2}-m_{1}))^{2}}{s_{1}^{2}+s_{2}^{2}}$b	by Eq(3)

​		   $=\frac{(w^{T}(m_{2}-m_{1}))(w^{T}(m_{2}-m_{1}))}{s_{1}^{2}+s_{2}^{2}}$	

​		   $=\frac{(w^{T}(m_{2}-m_{1}))((m_{2}-m_{1})^{T}w)}{s_{1}^{2}+s_{2}^{2}}$

​		   $=\frac{w^{T}S_{B}w}{s_{1}^{2}+s_{2}^{2}}$	

​		   $=\frac{w^{T}S_{B}w}{\Sigma_{n\in C_{1}}(y_{n}-m_{1})^{2}+\Sigma_{n\in C_{2}}(y_{n}-m_{2})^{2}}$	by Eq(5)

​		   $=\frac{w^{T}S_{B}w}{\Sigma_{n\in C_{1}}(w^Tx_{n}-m_{1})^{2}+\Sigma_{n\in C_{2}}(w^Tx_{n}-m_{2})^{2}}$	by Eq(1)

​		   $=\frac{w^{T}S_{B}w}{\Sigma_{n\in C_{1}}(w^Tx_{n}-w^Tm_{1})^{2}+\Sigma_{n\in C_{2}}(w^Tx_{n}-w^Tm_{2})^{2}}$	by Eq(4)

​		   $=\frac{w^{T}S_{B}w}{\Sigma_{n\in C_{1}}(w^Tx_{n}-w^Tm_{1})(w^Tx_{n}-w^Tm_{1})+\Sigma_{n\in C_{2}}(w^Tx_{n}-w^Tm_{2})(w^Tx_{n}-w^Tm_{2})}$	

​		   $=\frac{w^{T}S_{B}w}{\Sigma_{n\in C_{1}}w^T(x_{n}-m_{1})w^T(x_{n}-m_{1})+\Sigma_{n\in C_{2}}w^T(x_{n}-m_{2})w^T(x_{n}-m_{2})}$	

​		   $=\frac{w^{T}S_{B}w}{\Sigma_{n\in C_{1}}w^T(x_{n}-m_{1})(x_{n}-m_{1})^Tw+\Sigma_{n\in C_{2}}w^T(x_{n}-m_{2})(x_{n}-m_{2})^Tw}$	

​		   $=\frac{w^{T}S_{B}w}{w^{T}(\Sigma_{n\in C_{1}}(x_{n}-m_{1})(x_{n}-m_{1})^{T}+\Sigma_{n\in C_{2}}(y_{n}-m_{2})(y_{n}-m_{2})^{T})w}$

​		   $=\frac{w^{T}S_{B}w}{w^{T}S_{W}w}$	Eq(7)

#### (7%) 4. Show the derivative of the error function Eq (8) with respect to the activation $a_{k}$ for an output unit having a logistic sigmoid activation function satisfies Eq (9).

![image-20221031070625975](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20221031070625975.png)

Suppose $t_{n1}=t_{n}$, $t_{n2}=1-t_{n}$, $y_{n1}=y_{n}$, $y_{n2}=1-y_{n}$

Then $E(w)=-\Sigma_{n=1}^{N}\Sigma_{k=1}^{K}t_{nk}\ln y_{nk}$, where $K=2$

However we have $\frac{\partial E}{\partial y_{nk}}=-\frac{t_{nk}}{y_{nk}}$ (eq. a)

According to $y_k=\frac{e^{a_k}}{\Sigma_{j}e^{a_j}}$

We have $\frac{\partial y_k}{\partial a_j}=y_k(I_{kj}-y_j)$, where $I_{kj}=\{1, j=k; 0, \text{otherwise.}$ (eq. b)

Given eq. a and eq. b, we can compute 

$\frac{\partial{E}}{\partial{a_{nj}}}=\Sigma_{k=1}^{K}\frac{\partial{E}}{\partial{y_{nk}}}\frac{\partial{y_{nk}}}{\partial{a_{nj}}}=-\Sigma_{k=1}^{K}\frac{t_{nk}}{y_{nk}}y_{nk}(I_{kj}-y_{nj})=-\Sigma_{k=1}^{K}t_{nk}(I_{kj}-y_{nj})=-t_{nj}+\Sigma_{k=1}^{K}t_{nk}y_{nj}=y_{nj}-t_{nj}$. (eq. c)

Since eq. c, we have Eq (9)

#### (7%) 5. Show that maximizing likelihood for a multiclass neural network model in which the network outputs have the interpretation $y_k(x, w)=p(t_k=1 | x)$ is equivalent to the minimization of the cross-entropy error function Eq (10).

![image-20221031070650689](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20221031070650689.png)

$p(T|w_1,\dots,w_k)=\prod_{n=1}^{N}\prod_{k=1}^{K}y_k(x_n, w)^{t_{kn}}$

$E(w)=-\ln(p(T|w_1,\dots,w_k))$

​	  	 $=-\ln(\prod_{n=1}^{N}\prod_{k=1}^{K}y_k{(x_n, w)}^{t_{kn}})$

​	  	 $=-\sum_{n=1}^{N}\sum_{k=1}^{K}\ln y_k(x_n, w)^{t_{kn}}$

​	  	 $=-\sum_{n=1}^{N}\sum_{k=1}^{K}t_{kn}\ln y_k(x_n, w)$		Eq(10)