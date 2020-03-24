# ArcFace: Additive Angular Margin Loss for Deep Face Recognition

## Abstract
1.  在 Deep CNN 進行大規模的人臉辨識中，定義具有分辨能力的 loss function 是很大的挑戰
    -   Center Loss 將每個 data 與他們所屬 class 的中心點做距離來計算類內緊致度
    -   SphereFace 假設 FCNN 中最後一層的 transformation matrix 可以用來當作 class center 的 representation。並且利用向量之間的 angle 來計算 loss

2.  本篇 paper 中，我們提出 Additive Angular Margin Loss (ArcFace) 來取得人臉辨識中分辨力更強的 feature

## 1. Introduction
1.  DCNN 將 face image 映射到類內距離小、類間距離大的 feature representation
2.  DCNN 於人臉辨識有兩條主要研究，但兩者都有明顯的缺點
    -   soft-loss-based method
        -   當 identity number n 增加時，linear transformation matrix 的 size 也將會跟著放大
        -   學到的feature 在 close-set 分類表現不錯，但是在 open-set 的表現較差
    -   triplet-loss-based method
        -   在大規模的 data-set 中，iteration steps 會產生組合爆炸的情形
        -   semi-hard sample mining 為一問題

3.  許多嘗試改進表現的作法
    -   Wen et al. 首先嘗試 centre loss。但是當 class 數量增加時，更新 class center 會變得困難

4.  本篇論文提出 Additive Angular Margin Loss
    -   DCNN feature 和最後一層 transformation matrix 的內積就相當於 feature 和 normalization weight 的距離
    -   使用 arc cos 來計算 current feature 和 target weight 的角距離
    -   在 target angle 加入 angular margin，再以 cos 函數取回 target logits
    -   利用固定的 feature norm 來 re-scale 所有的 logits
    -   隨後的步驟則和 softmax loss 相同

5.  本篇論文作法優勢為
    -   Engaging
        -   ArcFace 取自測地線
    -   Effective
        -   達成了十項人臉辨識基準，包含影片資料集、大規模影像資料
    -   Easy
        -   程式碼簡潔
    -   Efficient
        -   ArcFace 只會加上幾乎可以忽略的計算複雜度
## 2. Proposed Approach
### 2.1. ArcFace
1.  最常使用的 classification loss function 為 soft-max loss
    <!--圖-->
    -   xi 屬於 R^d 空間為第i個 sample 的 deep feature, 屬於第 yi 個 class
    -   embedding feature 維度 d 在本篇論文中設為 512
        -   Wj 屬於 R^d 空間，記下 R^{dxn} 空間中 weight 矩陣 W 的第 j 行
        -   bj 屬於 R^n 空間，為 bias term
    
    -   batch size 為 N, class number 為 n

2.  傳統 softmax loss 在 deep face recognition 中有廣泛使用
    -   然而，softmax loss function 並沒有將feature embedding 作最佳化以達到 intra-class sample 高度 similarity 與 inter-class 高度 diversity

    -   造成在 intra-class 有像是姿勢、年齡等較多 variation，或是 large-scale test 的情況時表現會較差

3.  化簡情況
    -   將 bias fix 為 0 (bj=0)
        -   則 logit 為 Wj^Txi (內積)
        -   theta_j 為 weight Wj 和 feature xi 的夾角
    
    -   將 individual weight 的 norm 化為 1 以達到 l2 normalisation

    -   將 embedding feature xi 的 norm 作 l2 normalisation 並 re-scale 到 s

    -   這些在 feature 和 weight 上的 normalisation 將使 prediction 只會基於 feature 和 weight 之間的 angle
        -   學到的 embedding features 會分佈在以 s 為半徑的 hypersphere 上

<!--圖(2)-->

4.  由於 embedding features 分佈在hypersphere 上
    -   我們在 xi 和 Wyi 之間加上 angular margin penalty m
        -   提高 intra-class compactness 和 inter-class discrepancy

<!--圖(3)-->

5.  我們從 8個不同的 identities 上選出 face images
    -   每個 class 約有 1500 張 images 用以 train 2D feature embedding network
        -   使用 softmax 與 ArcFace loss
        -   可以得到更清晰的 decision boundary
<!--圖(超球面)-->

### 2.2. Comparison with SphereFace and CosFace
`Numerical Similarity`
1.  SphereFace, ArcFace 和 CosFace 為三種不同的 margin penalty
    -   可乘的 angular margin m1
    -   可加的 angular margin m2
    -   可加的 cosine margin m3

2.  從數值分析的角度，所有的 margin penalties 都會用 penalising target logit 的方式帶來 intra-class compactness 和 inter-class diversity

3.  從 target logit curves 看, 可以發現
    -   ArcFace 的 training 從 90 度左右開始並結束在 30 度
    -   影響 performance 的關鍵3個 factor 為 starting point, end point 和 slope
<!--斜率圖-->

4.  透過結合所有 margin penalties, 我們將 SphereFace, ArcFace 和 CosFace 實作在一個統一的 framework 上
    -   包含 m1, m2, m3 三個 hyper-parameters
    -   可以得到更好的表現

`Geometric Difference`
1.  ArcFace 在 numerical similarity 上比以往的 works 有更好的表現，同時也有更好的幾何性質
    -   ArcFace 有 constant linear angular margin, 落在整個 interval 之間
    -   SphereFace 和 CosFace 則只有 nonlinear angular margin

<!--margin 圖-->

2.  在 margin design 上, 於 model training 有 butterfly effect
    -   原本的 SphereFace 採取 annealing 最佳化策略
    -   防止在 training 開始時出現發散,joint supervision from softmax 被使用在 SphereFace
        -   減少 multiplicative margin penalty

### 2.3. Comparison with Other Losses
1.  loss function 可以基於 features 和 weight-vectors 的 angular representation
    -   可以設計 loss 來 enforce hypersphere 上的 intra-class compactness 和 inter-class discrepancy

`Intra-Loss`
-  透過減少 sample 和 ground truth center 之間的 angle/arc用以增加 intra-class compactness

`Inter-Loss`
-  透過增加不同 center 之間的 angle/arc 來增加 inter-class discrepancy

2.  Inter-Loss 是 Minimum Hyper-spherical Energy (MHE) 的特例
    -   hidden layers 和 output layers 都被 MHE regularised
    -   在 network 的最後一層可能也會結合 SphereFace loss 和 MHE loss

`Triplet-loss`
-   目標為放大 triplet 之間的 angle/arc margin

-   在 FaceNet 中, Euclidean margin 被用在 normalised feature 上

-   本篇論文將 triplet-loss 用在 feature 的 angular representation 上