# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation 

## Abstract
1.  Point Cloud 是一種重要的幾何資料結構
    -   由於此資料結構的格式較不規律，研究者多半將此資料轉換成 3D voxel grids 或是 images
    -   然而這種作法會讓 data 變地肥大

2.  本篇論文，提出一種 neural network `Point Net`
    -   直接使用 point cloud
    -   respect input points 的 permutation invariance
    -   可以用統一的架構應用在 object classification、scene semantics parsing、part segmentation

3.  以下將會分析並理解此 network 學到的以及為何此 network 對於 input 的 pertubation 和 corruption 能有 robust 的表現

## 1. Introduction
1.  典型的捲積架構需要高度 regular 的 input data format 來進行 weight sharing 或是 kernel optimization
    -   如 image grid, 3D voxels

2.  由於 point clouds 和 meshes 不是 regular format, 多數研究者會將它轉換成 regular 3D voxel 或是 collections of images 來送入 neural network
    -   然而，這樣的作法會讓 data 產生不必要的膨脹，同時會破壞一些 data 原有的 invariance

3.  本篇論文採用 PointNet
    -   以 point clouds 來作為 input representation
    -   可以避免掉 meshes 的 combinatorial irregularities 和 complexities

    -   同時，PointNet 會保有 input point set `invariant to permutation` 的性質，意即 Net 的運算會有對稱性

    -   更進一步，此 invariant 的特性甚至也適用於 rigit motions

4.  本篇論文的 PointNet 是統一的架構
    -   以 point clouds 作為 input
    -   output 出所有 input point 或是 point segment/part 的 class labels

    -   每個 point 都以座標(x,y,z)表示
        -   可能會再加上法線或是其他local、global的特徵

5.  本篇論文的關鍵為使用單一對稱函數 max pooling
    -   讓 network 學到一系列的 optimization functions/criteria 來選擇 informative points 並且 encode 出選擇的原因

    -   最後再以 fully connected layers 來 aggregate 學到的 optimal values 到 global descriptor
        -   shape classification (整體 shape)
        -   shape segmentation (per point label)

6.  本篇論文能夠容易地應用到 rigid, affine transformation
    -   因為 point transform independently
    -   可以在使用 PointNet 之前先對 data 作正規化來提升結果

7.  本篇論文提出了理論分析與實驗評估
    -   network 可以逼近任何 continuous set function
    -   network 學習將 point cloud 用 sparse set of key points 來 summarize
        -   大約是 object 的輪廓
    
    -   分析 PointNet 對 input point 有 perturbation、missing data、outliers 能有 robust 表現的原因

`Key Contribution`

1.  本篇論文設計出 deep net 架構，可用於 3D unordred point set

2.  本篇論文呈現出 Net 如何做到 3D shape classfication, shape part segmentation, scene semantic parsing task

3.  本篇論文提供實驗依據與理論分析
4.  本篇論文提供由 neurons 計算出的 3D 特徵並且以直覺的方式解釋它的表現

## 2. Related Works
`Point Cloud Features`

1.  point cloud 的 feature 多半設計以針對特定 tasks
    -   通常是 encode point 特定的統計性質
    -   設計上會讓 feature 對特定 transformation 保持不變性
    -   可分成 intrinsic, extrinsic, local, global

`Deep Learning on 3D Data`

1.  3D data 有多種 representations
    -   Volumetric CNNs
        -   使用 voxelized shapes 來進行 CNN
        -   3D convolution 在運算資源上的需要以及 data sparsity 會限制 reolution
    
    -   FPNN & Vote3D
        -   解決 sparse 的問題，但仍不適用於 large point cloud
    
    -   Multiview CNNs
        -   將 3D point cloud render 成 2D image 並且嘗試用 2D conv 來進行 classify

        -   能夠良好地進行 shape classification 以及 retrieval tasks

        -   然而對於 scene understanding 以及其他 3D tasks 如同 point classification、shape completion 仍有待延伸

    -   Spectral CNNs
        -   將 CNN 用在 meshes 上
        -   受限於 manifold meshes，non-isometric shapes 仍需延伸

    -   Feature-based DNNs
        -   將 3D data 利用傳統 shape feature 轉換成 vector 並用 fully connected net 來分類 shape
        -   feature 的 representation power 仍然存在疑慮

`Deep Learning on Unordered Sets`

1.  從 data structure 的角度來看，point cloud 是 unordered set of vectors
    -   大多數 deep learning 的 works 都是專注在 regular input representations，如同 sequences
    -   針對 point sets 的 deep learning 尚無太多著墨

2.  Oriol Vinyals et al 與此問題有相關研究
    -   read-process-write network
        -   使用 unordered input sets
        -   呈現出能夠排序數字的能力
    -   由於是應用在 NLP，因此約乏考慮 sets 的幾何性質

## 3. Problem Statement
1.  point cloud 可以用 3D point 的集合來表示
    -   {Pi | i=1,...,n}
    -   Pi 是每個點的 vector
        -   coordinate, color, normal 等 feature channel
        -   本篇論文以 coordinate 做討論

2.  task
    -   object classification
        -   input point cloud 為從 scene point cloud 的 pre-segmented sample 出來的

        -   network 會 output 出 k 個 score 給 k 個候選的 class

    -   semantics segmentation
        -   input 可以是 single object, sub-volume

        -   network 會 output nxm 個 score 給 n 個 points。每個 point 有 m 個 semantic sub-categories

## 4. Deep Learning on Point Sets
-   本篇論文的 network 架構是從 R^n 空間 point set 性質得到啟發
### 4.1 Properties of Point Sets in R^n
1.  input 為歐式空間中 points 的子集合，有以下三種性質
    -   Unordered
        -   不同於 image 的 pixel array 或 volumetric grid 的 voxel array, point cloud 沒有特定順序

        -   network 使用 N 個 3D points 必須對 N! 種排序有不變性

    -   Interaction among points
        -   points 有 distance metric
        -   points 不是 isolated，neighboring points 會形成 meanful subset
        -   model 要有辦法捕捉出 nearby points 形成的 local structure 以及 local structures 之間的 combinatorial interaction

    -   Invariance under transformations
        -   學到的 representation 必須對於特定 transformaion 有不變性
        -   例如平移和轉動不會影響到 global point cloud category 以及 points 的 segmentation

### 4.2 PointNe Architecture
1.  本篇論文的 full network 架構包括 classification network 以及 segmentation network。兩者 share 許多部份的結構。

2.  network 架構有3個關鍵 modules
    -   使用 max pooling layer 作為 symmetric funciton 
        -   從所有 points 中 aggregate information
        -   local and global information combination structure
        -   two joint alignment networks 來 align input points 和 point features

`Symmetry Funtion for Unordered Input`

1.  為了讓 model 對於 input 的排列能有不變性，有三種策略
    -   將 input 以 canonical order 做排序
    -   將 input 以 sequence 來 train RNN，但是對 data 以各種排列來做 data augment
    -   使用 simple symmetric funciton 來 aggregate information

-   symmetric function 將會以 n 個 vector 為輸入並且輸出一個新的 vector
    -   無視 input 的 order
    -   '+' 和 '*' 都是 symmetric 的 binary function

2.  在高維度空間中，sorting 並不是 solution
    -   如果 ordering strategy 存在，則將有一個 bijection map 存在高維空間和1d直線間。
    
    -   一般情況是無法靠著一個 mapping 就保持高維空間中的 spatial proximity 並做到 dimension reduce

3.  希望能夠透過以各個排列方式訓練 RNN 以達到讓RNN 可以對 input 順序有不變性
    -   然而，在大規模的 data 下， RNN 仍然沒辦法完全做到 robustness

4.  本篇論文的方法是要用 symmetric funciton g 逼近定義在 point set 的general function f 上。
    -   f 以 power set 2^(R^N) 為定義域，map 到 R
        -   R^N 空間中每個點都有選和不選，形成子集合的集合 2^(R^N)
    
    -   h 是 Multi layer perceptron network，g 是單變數函數的 composition 以及 max pooling function
        -   透過收集多個 h，可以學到多個 f 的 feature 並且捕捉 point set 的性質

`Local and Global Information Aggregation`

1.  上述函數將會 output 出 vector (f1,f2,...,fk)
    -   此 vector 是 input set 的 global signature
        -   可以利用 SVM/MLP 來做分類
    
    -   segmentation 還需要 local knowledge

2.  將 global feature 連接到 per point feature 形成 combined point features
    -   此時就同時有 local 和 global 的資訊

3.  藉由此調整，model 能夠 predict per point，透過
    -   local geometry & global semantics

`Joint Alignment Network`

1.  point cloud 的 semantics labeling 必須對於特定 transformation 有不變性
    -   例如 rigid transformation

2.  將整個 input set align 到 canonical space 再做 feature extraction 是一個 solution
    -   Jaderberg et al 利用 spatial transformer 以 sampling 和 interpolation 來 align 2D images

3.  本篇論文中
    -   以一個 mini-network (T-net) predict 一個 affine transformation matrix 並以此 matrix 對 input point 的座標進行 transformation

    -   也能夠延伸到 feature space 的 alignment
        -   feature space 的維度比 spatial transform matrix 更高，在 optimization 較為困難，因此會加上 regularization term
            -   Lreg = ||I - AA^T||^2_F
            -   A 為 mini-network predict 出的 feature alignment matrix
            -   希望 A 能夠保持 orthogonal

### 4.3 Theoretical Analysis
`Universal approximation`

1.  本篇論文的 network 的 universal approximation 能力
    -   set function 的 continuity 性質必須達到 small perturbation 不會帶來 classification 及 segmentation scores 的大影響

2.  令 X = {S:S 包含於 [0,1]^m 且 S有n個元素}
    -   f:X->R 為一個連續的set function
        -   遵循 Hausdorff distance，任意兩個X內的點S,S'只要足夠近，則f(S)和f(S')也可以任意接近
    
3.  以下 Theorem 敘述
    -   只要 max pooling layer 的 neurons 數 K 夠大，本篇論文的network可以任意逼近 f

`Theorem 1`

1.  令 f:X->R 是一個連續的 set function 
    -   Hausdorff distance dH(.,.)
    -   對任意正數 epsilon, 都能找到一個連續函數 h 和對稱函數 g(x1,...,xn)=(gamma)(MAX) 使得對任何 S 屬於 X
        -   f(S) 和 g(h(x)) 的距離小於 epsilon

2.  意即 network 的 worst case 可以學習將 point cloud 轉換成 volumetric representation
    -   將 space 分割成等大的 voxels
    -   然而 network 有更 smart 的策略

`Bottleneck dimension and stability`
1.  從實驗上發現，本篇論文的 network 表示能力明顯受到 max pooling layer 的 dimension 影響

2.  定義 u=MAX{h(xi)}, xi屬於S 為 f 的 sub-network
    -   將 [0,1]^m 中的 point set mapping 到 K-dimensional vector
    -   Theorem 2 會敘述 input set 中 small corruption 以及額外的 noise points 不會影響到 network 的 output

`Theorem 2`
1.  令 u:X->R^K 使得 u=MAX{h(xi)}, xi屬於S 且 f=(gamma)(u)
    - (a) 對所有的S, 存在一個 Cs,Ns 包含於X
        -   f(T)=f(S), 若 Cs包含於T包含於Ns
    
    - (b) Cs的元素數量小於等於K

2.  此定理的解釋
    -   (a) 表示, 對任何S
        -   只要 Cs 的 point 都有保留住，就能有和 f(S) 相同的 output
        -   即使多出了 Ns 的 noise point，也能有和 f(S) 相同的 output
    
    -   (b) 表示 Cs 只有 bounded number of points
        -   因此 f(S) 其實可以被元素數量少於等於 K 的 finite subset Cs 包含於 S 確定
        -   因此稱 Cs 為 S 的 critical point set, K 為 f 的 bottleneck dimension

3.  此 robustness 可從 machine learning 中的 sparsity principle 中取得類比
    -   network 學到從 sparse set of key point 中 summarize shape