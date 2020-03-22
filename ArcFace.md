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
### 2.2. Comparison with SphereFace and CosFace
### 2.3. Comparison with Other Losses
## 3. Experiments
### 3.1. Implementation Details
### 3.2. Ablation Study on Losses
### 3.3. Evaluation Results
## 4. Conclusions
