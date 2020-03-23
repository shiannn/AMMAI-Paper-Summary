# VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection

## Abstract
1.  在 3D point clouds 中進行準確的 object detection 一直是許多應用中的中心問題
    -   自動導航, 管家機器人, 擴增實境與虛擬實境

2.  要為 sparse LiDAR point 設計一個能讓 region proposal network (RPN) 使用的 interface, 將需要 hand-crafted feature representations
    -   bird's eye view projection

3.  本篇論文中，移除了 3D point clouds 所需的特徵工程並且提出 VoxelNet
    -   此 Net 為一個 3D detection network, 可以在單一 stage 統一 feature extraction 以及 bounding box prediction, 為一個 end-to-end trainable network

    -   VoxelNet 會將 point cloud 分成相等的 space 3D voxels
        -   利用 voxel feature encoding (VFE) layer 將每個 voxel 中的 points encode 成統一的 feature representation

        -   point cloud 可被 encode 成 descriptive volumetric representation 並且利用 RPN 來進行 detection
    
    -   在 KITTI 的 car detection benchmark 中可以看出 VoxelNet 超越 LiDAR based 的 3D detection methods

    -   此 network 也學到了對不同幾何形狀的物體做出 discriminative representation 的能力，可以對於行人、騎士作 3D detection

## 1. Introduction
1.  不同於 images ，LiDAR point clouds 較為 sparse 且 point 的密度分佈非常不一
    -   原因是 range of sensors, occlusion, relative pose
    -   通常必須 handcraft feature representation 來處理這個問題
        -   project point clouds 到 perspective view 並且 apply image-based feature extraction

        -   rasterize point cloud 到 3D voxel grid 並且 encode 每個 voxel 成 handcrafted features
        
        -   然而這些作法都存在問題。像是無法有效表達 3D 形狀資訊，或是沒辦法保有 detection 需要的 invariances 性質
    
    -   recognition 和 detection 最大的突破就是將 hand-crafted feature 改為 machine-learned features

2.  Qi et al. 提出了 PointNet, 後又更進一步有人提出能夠於不同 scales 學到 local structure 的版本
    -   這兩個 approach 會用所有 input points 對 feature transformer networks 做訓練
        -   一般 LiDAR 的 points 數約有 100k, 需要高運算力與儲存資源

        -   本篇論文的目的為使 3D feature learning networks 能夠用於大量 points 與 3D detection tasks

3.  Region proposal network 為高度最佳化方法
    -   需要 dense data 且為 tensor structure (image, video)

    -   因此本篇論文會處理 point set feature learning 以及 RPN 3D detection task 的差距

4.  VoxelNet 同時進行學習 point cloud 的 feature 與預測 3D bounding boxes
    -   novel voxel feature encoding layer (VFE) 能使 voxel 內的 point 做到 inter-point interaction

    -   疊起多層 VFE Layers 可以學到複雜的 feature 以取得 local 3D shape 資訊

    -   3D convolution 聚集 local voxel features, 將 point cloud transform 成 high-dimensional volumetric representation

    -   最後再以 RPN 使用 volumetric representation 取得 detection result

5.  此演算法獲益於 sparse point structure 以及 voxel grid 的平行運算
    -   將 VoxelNet 以 bird's eye view detection 以及 full 3D detection tasks 作 evaluate, 在 KITTI 的 benchmark 上都超越 LiDAR based 3D detection method

### 1.1 Related Work
1.  hand-crafted features 在有豐富詳細的 3D shape information 時能有不錯的效果，但缺點是
    -   難以適應複雜的 shape 與場景
    -   需要有 invariance

2.  image 能夠提供相當多 texture information
    -   許多演算法能從 2D images 推論出 3D bounding box，但正確率多半會受到 depth estimation 的限制

3.  許多 LIDAR based 3D object detection 使用 voxel grid representation
    -   在每個 non-empty 的 voxel 中，用所有的 point 計算出 6 個統計量
    -   加上 local statistics 形成該 voxel 的 represent
    -   對 3D voxel grid 作 binary encoding
    -   用 multi-view representation
    -   將 point clouds 投影到 image 上並且進行 image-based feature encoding

    -   結合 image 和 LiDAR 的 multi-modal fusion method
        -   對於小物體有更好的準確度
        -   但是需要額外的 camera 同時與 LiDAR 運作以及校準，對 sensor 更加敏感

### 1.2 Contributions
1.  以 end-to-end 的 deep 架構作 point-cloud-based 3D detection, 直接在 sparse 3D points 上運作以避免特徵工程在 information 上的瓶頸

2.  利用 voxel grid 的平行運算以及 sparse point structure 取得運算效率

3.  於 KITTI benchmark 上進行過實驗

## 2. VoxelNet
1.  在本 section 將會介紹 VoxelNet 的架構
    -   loss function
    -   efficient algorithm

### 2.1 VoxelNet Architecture
1.  VoxelNet 有3個blocks
    -   Feature learning network
    -   Convolutional middle layers
    -   Region proposal network
#### 2.1.1 Feature Learning Network
`Voxel Partition`

1.  在 3D 空間中有等分開來的space voxel
2.  Z, Y, X 3個軸上有 D, H, W 的 range
    -   每個 voxel size 為 vD, vH, vW
    -   3D voxel grid 的 range 就是 
        -   D'=D/vD, H'=H/vH, W'=W/vW

`Grouping`

1.  根據 points 落在的 voxel 來分組
2.  LiDAR 的 points 分佈相當不均勻
    -   在分組後各個 voxel 內的 points 數量不一
`Random Sampling`

1.  LiDAR 的 point cloud 含有 100k 個 point 
2.  直接處理所有 points 會帶來運算資源的負擔並且無法避免 points 分佈不一帶來的 detection bias
3.  因此會選定一個固定數字 T
    -   隨機從 voxels 中 sample 出T個 points
        -   減緩運算負擔
        -   減少 voxel 之間的 imbalance
        -   減少 sample bias
        -   增加 training data 的多樣性

`Stacked Voxel Feature Encoding`

1.  一個 Voxel 可以用 points 的集合來表示
    -   V = {pi = [xi,yi,zi,ri]}, i=1,...,t
    -   pi 包含了 point 的位置資訊以及 reflectance ri

    -   先計算 V 中的 point 的質心位置並且把每個 point 相對質心的位置連接到 pi 作為 feature，形成7維向量

2.  將7維向量作為 input 輸入 FCN 中
    -   aggregate point feature 的資訊來encode 出 voxel 中包含的 surface shape
    -   FCN 包含 BN, ReLU

3.  使用 MaxPooling 對所有 V 中的 fi 計算一個 locally aggregated feature f'，再將 f' 連接到每個 fi 以取得 output feature set Vout

4.  VFE-i(cin,cout) 表示第i層 VFE layer
    -   將cin維度 transform 到cout維度
    -   其中 linear layer 的 transform matrix 有 cin x (cout/2) 的 size。再透過 point-wise concatenation 形成 cout 維度

`Sparse Tensor Representation`

1.  取得的 list of voxel-wise feature 可以表示為 sparse 4D tensor
    -   C x D' x H' x W' 大小

2.  雖然有將近 100k 的 points，但 90% 的 voxel 都是空的，因此利用 non-empty voxel feature 來做 representation 可以節省 memory usage 以及 Back propogation 的 computation cost

#### 2.1.2 Convolutional Middle Layers
1.  ConvMD(cin,cout,k,s,p) 表示M維度 convolution 運算
    -   cin 和 cout 為 input 和 output 的 channel 數
    -   k 為 kernel 數、s 為 stride 數、p 為 padding size 數，均為 M 維

2.  每個 ConvMD 都會進行 3D convolution 以及 BN layer 和 ReLU layer
    -   convolution middle layers 聚集 voxel-wise 的特徵

#### 2.1.3 Region Proposal Network
1.  Region Proposal Network 在現今 object detection 與 feature learning network、Convolutional middle layer 結合以形成 end-to-end trainable pipeline

2.  RPN 以 convolutional middle layers 的 feature map 作為 input
    -   network 一開始有3個 fully convolutional layer 組成的 block 
        -   每個 block 的第一個layer 會以 stride size=2 的 convolution 將 feature map downsample 成一半大小

        -   再以 stride size=1 的 convolution 作 q 次
    
    -   進行 BN, ReLU operations

    -   利用 Deconvolution 把每個 block upsample 到固定的 size 並連接成 high resolution feature map。會依learning target map到：
        -   probability score map
        -   regression map

### 2.2 Loss Function

1.  對於 bounding box 可以以下數學描述
    -   令 {ai^pos} i=1,...,Npos 為 positive anchor
    -   {ai^neg} i=1,...,Nneg 為 negative anchor

    -   可將 3D ground truth box 參數化為 (xc,yc,zc,l,w,h,theta)
        -   xc,yc,zc 為中心位置
        -   l,w,h 為 bounding box 的長寬高
        -   theta 為以Z軸為轉軸的 yaw rotation

2.  可以定義出 residual vector (dx,dy,dz,dl,dw,dh,dtheta)
    -   dx = xc^g - xc^a/d^a, dy = yc^g - yc^a/d^a, dz = zc^g - zc^a/h^a
        -   d^a 為 bounding box 底面的對角線長度，意即將位置誤差對 bounding box 的大小做標準化
    -   dl = log(l^g/l^a), dw = log(w^g/w^a), dh = log(h^g/h^a)
    -   dtheta = theta^g - theta^a

3.  Loss function 如下定義
    <!-- loss function 圖-->
    -   利用 alpha, beta 來作為重要性係數
        -   Lcls 為 binary cross entropy loss
            -   pi^pos 為 positive anchor ai^pos 和 negative anchor aj^neg softmax output
        -   Lreg 為 regression loss
            -   ui 為 regression output
        (SmoothL1 function)
    
### 2.3 Efficient Implementation
1.  GPU 對於處理 dense tensor structures 有經過優化
    -   本篇論文將 point cloud 轉換成 dense tensor structure

2.  以 K x T x 7 的 tensor structure 來儲存 voxel input feature buffer
    -   K 為 non-empty voxel 的最大數量
    -   T 為 每個 voxel 中 point 數的最大數量
    -   7 為 每個 point 的 encoding dimension

3.  point cloud 中的每個 point，檢查對應的 voxel 是否存在 (O(n))
    -   如果尚未存在，則初始化一個voxel 並且將其座標存入 coordinate buffer
    -   如果已經存在
        -   若 voxel 中少於 T 個 points，則將 point insert 到此 voxel
        -   否則忽略此 point

## 3. Training Details
-   VoxelNet Implement detail and training proceure
### 3.1 Network Details
-   本篇論文的實驗以 LiDAR specification 於 KITTI dataset
`Car Detection`

1.  考慮 point clouds range 在 [-3,1]x[-40,40]x[0,70.4] (Z,Y,X)
    -   忽略超出邊界的 points
    -   vD = 0.4, vH = 0.2, vW = 0.2
    -   D' = 10, H' = 400, W' = 352 為 voxel grid 的 size
    -   T = 35 為每個 voxel 中隨機 sample 出 point 的最大值

2.  使用 VFE layer
    -   VFE-1(7,32) 和 VFE-2(32,128)
    -   FCN 將 VFE-2 映射到 R^128

-   整個 feature learning net 生成 128x10x400x352 的 sparse tensor

3.  Region Proposal Network
    -   Conv3D(128,64,3,(2,1,1),(1,1,1))
    -   Conv3D(64,64,3,(1,1,1),(0,1,1))
    -   Conv3D(64,64,3,(2,1,1),(1,1,1))

4.  只使用一種 anchor size
    -   bounding box 長寬高 l^a = 3.9, w^a = 1.6, h^a = 1.56
    -   中心點 zc^a = -1.0
    -   只有兩種旋轉角度 0 和 90 度

5.  Anchor matching
    -   IoU 超過 0.6 或有最高的 IoU
    -   IoU 低於 0.45
    -   IoU 介於 0.45 和 0.6 之間則忽略
        -   alpha = 1.5, beta = 1

`Pedestrian and Cyclist Detection`

1.  input 的 range 分別落在 Z,Y,X 三個軸上的 [-3,1] x [-20,20] x [0,48] 區間
    -   使用和 car detection 相同的 voxel size
    -   D = 10, H = 200, W = 240 (voxel座標)
    -   設 T = 45 來取得更多 LiDAR points 以捕捉 shape information

2.  feature Network 和 convolutional middle layers 皆與 car detection task 的 network 相同

3.  RPN 則在 block1 做出細微修改:將第一個 2D convolution 的 stride size 從 2 改為 1
    -   此舉可以讓 anchor matching 的 resolution 更好，利於捕捉行人與騎士

4.  anchor為
    -   中心位置 zc^a = -0.6
    -   0 或 90 度兩種旋轉
    -   size 
        -  行人 l^a = 0.8 w^a = 0.6 h^a = 1.73
        -  騎士 l^a = 1.76 w^a = 0.6 h^a = 1.73
    
    -   anchor 為 positive, 若
        -   與任一個 ground truth 的 IoU 為最高或超過 0.5
        -   與所有的 ground truth 的 IoU 都低於 0.35
        -   介於 0.35 和 0.5 之間則忽略

5.  使用 stochastic gradient descent
    -   前 150 epochs 的 learning rate 為 0.01
    -   最後 10 個 epochs 的 learning rate 減為 0.001
    -   batch size 16 point clouds

### 3.2 Data Augmentation
1.  少於 4000 個 Training point clouds，在本篇論文的 network 無法避免 overfitting
    -   引入三種型式的 Data Augmentation
    -   Augmented Data 會以 on-the-fly 的方式生成，無需佔用額外儲存空間

2.  定義集合 M = {pi = [xi,yi,zi,ri] 於 R^4 空間}, i=1,...,N 為整個 point cloud
    -   包含 N 個 points

3.  定義 3D bounding box bi 為 (xc,yc,zc,l,w,h,theta)
    -   xc,yc,zc 為 center locations
    -   l,w,h 為長寬高
    -   theta 為以 Z 軸為轉軸的旋轉角度

4.  定義落在 bounding box 裡的 point set 為
    omega_i = {p|x屬於[xc-l/2, xc+l/2],y屬於[yc-w/2, yc+w/2],z屬於[zc-h/2, zc+h/2], p 屬於 M (整個point cloud)}
    -   p = [x,y,z,r] 為落在 M 中的 point

5.  第一項 Data Augmentation 對每個 ground truth 3D bounding box 作 perturbation
    -   對 bounding box bi 連同其內的 point 集合 omega_i 作旋轉 
    -   角度取 [-pi/10, pi/10] 區間內均勻分佈
    -   再以 dx,dy,dz 對 bounding box bi 連同其內的 point 集合 omega_i 作平移
    -   位移量取 Gaussian distribution (平均為0, 變異數為1)
    -   完成 perturbation 之後再檢查所有 bounding box 是否有不合物理情況的衝突

6.  第二項 Data Augmentation 對每個 ground truth boxes bi 以及全體 point cloud M 作 global scaling
    -   對所有 M 中的 points 座標以及 bounding box bi 乘以一個 [0.95, 1.05] 中均勻分佈 sample 出來的數
    -   提升 model 對 various size and distance 的抵抗力

7.  第三項 Data Augmentation 會對整體 bounding boxes 以及 point cloud M 對 Z 軸公轉。
    -   公轉角度為 [-pi/4, pi/4] 間的 uniform distribution
    -   此公轉希望能夠模擬交通工具轉彎的情況