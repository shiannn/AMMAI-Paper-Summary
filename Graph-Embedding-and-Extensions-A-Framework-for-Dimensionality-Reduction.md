# Graph Embedding and Extensions: A Framework for Dimensionality Reduction
## Abstract
1.  Graph Embedding 可用以描述 dataset 的幾何/統計性質
2.  利用 constraints，可以讓 model 避免掉特定的幾何/統計性質
3.  利用 intrinsic graph 來描述 intra-class compactness 以及 penality graph 來對 margin point 形塑 interclass separability，得到比 LDA 表現更加的 MFA

## 1. Introduction
1.  本篇論文提供 graph embedding 作為一種統一所有 dimensionality reduction 方法的框架
    -   加上 linearization, kernelization, tensorization 增加 representation 的 ability

2.  本篇論文以 graph embedding framework 作為一個 platform，提出新的 dimensionality reduction algorithm
    -   focus on LDA 的變形：MFA
        -   available 的 projection direction 比 LDA 更多
        -   不需要對 data distribution 做出假設，更加 general
        -   interclass margin 比 LDA 的 interclass scatter 更能表達 separability

## 2. Graph Embedding: A General Framework For Dimensionality Reduction
1.  眾多 Dimensionality reduction 的目的皆為取得 representation 以進行隨後的 classification 工作
2.  本篇論文將敘述統一的 framework 以進行這些方法

### 2.1 Graph Embedding
1.  data 的 sample set 可以用一個 matrix X 表示
    -   X = [x1,x2,...,xN], xi 屬於 R^m
    -   此處 N 為 sample 的數量，m 為 feature 的維度

2.  class label ci 屬於 {1,2,...,Nc}
    -   Nc 為 class 的數量

3.  dimensionality reduction 的目的即為找尋一個 function F:x->y
    -   降低 feature 維度
    -   x 屬於 R^m, y 屬於 R^m', 其中 m' << m

4.  此處定義一個 graph
    -   將 sample set 定義為 graph vertex
    -   將 sample set 之間的 similarity 定義為 distance matrix W 的值
        -   Guassian
        -   Euclidean 等 distance metric
    -   定義 L = D - W
    -   其中 D 為對角矩陣，Dii = sum_(i!=j)(Wij)

5.  Graph Embedding 即為找到一組 y = [y1,y2,...,yN]
    -   yi 為 vertice xi 的 low-dimension representation
    -   y* = arg min_(y^TBy=d) sum(|yi-yj|^2Wij) = arg min_(y^TBy=d) y^TLy

    -   從上式可以看出，xi 和 xj 的相似度愈高，yi 和 yj 的 distance 就必須要愈小來 minimize objective function

6.  Graph Embedding 僅僅將 vertice 表現成 representation。對於其他 point
    -   Linearization
        -   可使用 linear projection y = X^Tw，w 為 unitary projection vector
    
    -   Kernelization
        -   利用 Kernel trick 將 data point map 到高維度的 Hilbert space，並在那進行 linear algorithm
            -   只需要計算 inner product k(xi,xj) = phi(xi)*phi(xj)
    
    -   Tensorization
        -   當 object 以 high-order structure 來表示時，僅僅 transform 成 vector 可能會導致 curse of dimensionality problem
        -   將 object encode 成任意 order 的 tensor

### 2.2 General Framework for Dimensionality Reduction
1.  PCA 為往最大化 variance 的方向進行 projection
    -   w* = arg min_(w^Tw=1) w^TCw
    -   C = 1/N sum_i^N (xi-xm)(xi-xm)^T = 1/N X(I-1/Nee^T)X^T
        -   C 其實為 covariance matrix
        -   xm 為所有 sample 的平均
        -   I 為 identity matrix

    -   因此 PCA 的 intrinsic graph 其實是將所有 weight 相同的 data pairs 連接在一起

2.  LDA 找尋能夠 minimize intraclass 和 interclass 的 ratio 的投影方向
    -   w* = arg min_(w^TS_Bw=d)w^TS_Ww = arg min_w(w^TS_Ww)/(w^TS_Bw) = arg min_w(w^TS_Ww)/(w^TCw)
    -   S_W = sum_i^N(xi-xm^ci)(xi-xm^ci)^T = X(I-sum_(c=1)^N(1/nc)e^ce^cT)X^T 
    -   S_B = sum_c^N nc(xi-xm^ci)(xi-xm^ci)^T = NC-S_W

    -   此處，xm^c 為第 c 個 class 的平均
    -   e^c 為 N 為向量
        -   e^c(i) = 1, c = ci
        -   0, otherwise
    
    -   因此，LDA 的 intrinsic graph 其實是將相同 class label 的 pair 以 class 大小的反比作為 edge weight 連起來
## 3. Marginal Fisher Analysis
1.  本篇論文所提出的 Graph Embedding Framework 除了包含眾多有名的 dimensionality reduction algorithms，同時也能用來設計新的 algorithms

2.  本篇論文將設計出新的 algorithm 以避開 Linear Disciminant Analysis 在 data distribution 和 projection directions 上的限制

### 3.1 Marginal Fisher Analysis
1.  LDA 假設每個 class 中的 data points 是以 Gaussian distribution 在做分佈，通常不會符合現實情況
    -   少了這個假設，interclass scatter 沒辦法很好地描述不同 class 間的 separability

2.  本篇論文利用 Graph Embedding 提出 Marginal Fisher Analysis
    -   利用 intrinsic graph 描述 intraclass 的 compactness
        -   每個 sample 都和同一個 class 中 k1 個最近的 neighbor 以 edge 相連
    -   利用 penalty graph 描述 interclass 的 separability
        -   在不同 class 之間，margin 上的 point pair 以 edge 相連
    
3.  演算法流程
    -   PCA projecyion
        -   先將 data set 投影到 PCA subspace，保留 N-Nc 維
    
    -   Constructing intraclass compactness 和 interclass separability graphs
        -   以每個 sample 點作為 vertex
        -   在 adjacency matrix W 上，如果 sample xj 是 xi(或者方向反過來) 在同一個 class 中最近的 k1 個 neighbor，則令 Wij = 1
        -   在 penalty matrix Wp 上，如果 sample xj 是 xi 在不同 class 上最近的 k2 個 neighbor，則令 Wpij = 1
    
    -   Marginal Fisher Criterion
        -   w* = arg min (w^TX(D-W)X^Tw)/(w^TX(Dp-WpX^Tw))
    
    -   輸出最後的 linear projection direction
        -   w = Wpca x w*

### 3.2 Kernel Marginal Fisher Analysis
1.  kernel trick 經常被用來增加 linear supervised dimensionality reduction 的 separation ability

2.  若 kernel function 為 k(xi,xj) = phi(xi)*phi(xj)
    -   Kernel Gram matrix 為 Kij = K(xi,xj)
    -   projection direction 為 w = sum_i^N(alpha*phi(xi))
    -   alpha* = arg min (alpha^TX(D-W)X^T*alpha)/(alpha^TX(Dp-WpX^T*alpha))

3.  當有新的 data point x，可以利用 optimal direction 得到
    -   F(x,alpha*) = lambda* sum_i^n(alpha_i* *k(x,xi))
    -   lambda = (alpha*^TKalpha*)^(-1/2)
