# Deep Learning for Understanding Faces

## Abstract
1.  現今的 deep convolution neural networks (DCNNs) 在 object detection/recognition 上的表現相當成功
    -   non-linear mapping on images and class labels
    -   large annotated data

2.  deep learning 同時也增進
    -   機器理解 faces
    -   face detection, pose estimation, landmark localization

3.  本篇論文將會討論 automatic face recognition system 的 modules

## What we can learn from faces
1.  face recognition 的目標為盡可能 extract 出多一些資訊
    -   location, pose, gender, ID, age, emotion
    -   應用於監視器, 智慧車輛, 快速交易

2.  本篇論文中, 呈現現代自動 face identification 和 verification system
    -   基於 deep learning
    -   通常使用3個 modules
        -   face detector 用以圖像或影片中的 localize faces 
            -   必須能夠兼顧不同的 pose, illumination, scale
            -   bounding box 包含最少程度的 background

        -   fiducial pointer detector 用以 localize 重要的 facial landmark
            -   eye center, nose tip, mouth corners
            
            -   使用這些 facial landmark 來將 face 與 normalized 的 canonical coordinates
                -   去除 in-plane rotation 和 scaling
        -   feature descriptor 來從 identity 的 face 上 encode 資訊
            -   給定兩個 face representations, 相似度運用 metric 計算出來

3.  自 1990 年代，face identification/verification 的演算法似乎運作良好
    -   然而當 data 在 pose, illumination, resolution, expression, age 有大量的variation 的話, 表現將下降
    -   監視器 video 的解析度需要更 fast 和 robust

4.  解決這些挑戰
    -   使用 Deep Learning 計算 facial 的 features
        -   DCNN (ImageNet Large Scale Visual Recognition Challenge)
        -   典型的 DCNN 為多層 Convolution Layers 加上 rectified linear unit activation function (RLU)
    
    -   現行 DCNN 在 face detection, fiducial point localization, face recognition 以及 face verification 上都有良好的表現
        -   一大原因為大規模的 uncontrained face data 加上標註
            -   CASIA-WebFace, Mega-Face, MS-Celeb-1M, and VGGFace 可用於 face recognition

            -   WIDER FACE 可用於 face detection
        
        -   大量的 training data 可以表示出 pose, illumination, expression, occlusion 的 variation
            -   使 DCNN 可以對這些 variation 有 robust 的表現並抽出有助於 task 的 features
## Face detection in unconstrained images
1.  Face detection 在 face recognition 的 pipeline 中扮演了重要角色, 且是 face recognition system 的第一步
    -   Face detection 找出所有 face 並且 return 所有 bounding box 的座標

2.  unconstrained face detection 中有一項對傳統 feature (HOG, Haar wavelets)的挑戰
    -   在不同的 resolution, illumination, expression, skin color, occlusions, cosmetic conditions 下不會捕捉 salient facial 資訊

    -   瓶頸通常是出現在 features 上而不是 classifiers, 但現在可以使用 DCNN 來抽取 features

    -   DCNN-based 的 method 可以分成兩類
        -   region based
        -   sliding window
### Region based
1.  region-based 的方法生成多個 object proposal (每張照片 2000 個)
    -   DCNN 用以分類這些 object proposal 看看有無包含 face
    -   一般 region-based face detector 都會包含現成的 object proposal generators
        -   例如 Selective Search
    -   再跟隨 DCNN 以分類看這些 proposals 有無 face
        -   例如 HyperFace, All-in-One Face
### Faster R-CNN
1.  最近代的 face detector 使用 faster R-CNN 來生成 proposals 並且分類出 faces
    -   可以同時為每個 face proposal regress bounding-box 的座標

2.  Li et al. 提出一個 multitask face detector
    -   使用 faster R-CNN 的 framework, 由 DCNN 和 3D mean face model 構成
        -   3D mean face model 用以提升 Region proposal network 在 fiducial detection 的 performance

3.  Chen et al. 提出 face detector
    -   訓練一個 multitask RPN 來取得 face, fiducial detection 並且生成 face proposals

    -   face proposals 之後就會以 detect 到的 fiducial points 作 normalized 並以 DCNN 作 face classifier

4.  Najibi  et  al. 提出 Single-Stage Headless face detector
    -   在 VGGNet 上使用 RPN, 未使用任何 fully connected layers

5.  缺點：
    -   複雜的 face 可能不會在任何 object proposal 中被捕捉到
    -   額外的 proposal 生成會增加運算時間

### Sliding-window based
1.  Sliding-window based method 會在每一種scale 的 feature map 內的每一個位置都計算一個 face detection score 以及 bounding-box 座標
    -   這個過程可以用 sliding window 的方式操作 convolution operation 
    -   多個 scale 通常以 image pyramid 的方式執行

2.  DCNN-based 的 face detectors 使用 sliding-window
    -   DP2MFD, DDFD

3.  Li et al. 提出 cascade 架構
    -   基於 DCNN, 於多個解析度上運作
    -   快速 reject 掉低解析度的背景物件
    -   只 evaluate 高解析度上的少數候選者
### Single-shot detector
1.  Liu et al. 提出 single-shot detector(SSD) 以解決 object detection
    -   屬於 sliding-window-based detector
    -   使用 DCNN 內部的 pyramid structure 而不是 image pyramids at different scale

    -   DCNN 的中間層會 pool 出不同 scale 的 feature 並用在 object classification 和 bounding-box regression 上

2.  Yang et al. 提出 ScaleFace
    -   在多個 layer 上偵測不同 scale 的 face 並且在最後混合結果

3.  FDDB data set 常用於 unconstrained face detection 的 benchmark
    -   從 yahoo 的新聞上取出 2845 張 images, 包含 5171 個 faces

4.  MALF dataset 包含 5250 張高解析度 images, 包含 11931 個 faces
    -   從 flick 上收集的 images
    -   包含許多 variation
        -   occlusion, pose, illumination

5.  WIDER Face data set 包含 32203 images
    -   50% 的 samples 用以訓練, 10% 的 samples 用以 validation
    -   這些 data set 包含 faces
        -   pose, illumination, occlusion, scale 的 variation
    
    -   人群中的 small face 仍然是挑戰

6.  Hu et al. 提出 method
    -   contextual information 對於偵測 tiny faces 相當重要
    -   捕捉 DCNN 中 low-level features 和 high-level features 中的 context 來偵測 tiny faces

7.  傳統方法, 本篇論文推荐讀者
    -   traditional cascade-based method
    -   deformable part-based model (DPM)

8.  在 video-based 的 face recognition 上
    -   需要進行 face association 以捕捉出每個 subject 的 face tracks
    -   推荐讀者閱讀[12]
    
## Finding crucial facial keypoints and head orientation
1.  Facial keypoints 在 face recognition 和 verification tasks 上仍然是重要的 preprocessing
    -   如 eye centers, nose tip, mouth corners
    -   可以將 face align to canonical 座標, 此 face normalization 能幫助 face recognition 和 attribute detection

2.  在 pose-based face analysis 需要做 Head-pose estimation

3.  現存的 facial keypoint localization 演算法都是使用 model-based 或是 cascaded regression-based 的方法

-   許多方法被發展出來以達成 face detection 上的 multitask learning (MTL)
    
        -   同時訓練 face-   Wang et al.  detection task 和 correlated task 如 facial keypoint estimation
        
        -   MTL 可以幫助 network 產生 synergy(協同作用) 並學到更 robust 的 feature
            -   因為 network 得到了額外的 supervision
            -   例如在 keypoint 中得到的 eye center 的資, nose tip訊可以幫助 network 確定 face 的結構[36]提出了 survey
        -   traditional model-based method
            -   active appearance model(AAM)
            -   active shape model (ASM)
            -   constrained local model (CLM)
            -   supervised descent method (SDM)
    
    -   Chrysos et al. [37] 提出 fiducial point tracking 於 video
        -   使用傳統 fiducial detection methods
    
    -   本篇論文將會簡述基於 DCNN 的 fiducial detection methods

### Model based
1.  model-based 方法, 如 AAM,ASM,CLM
    -   在訓練時 learn 出 shape model 並在測試時將其用以 fit 新的 face
    -   然而, learned model 缺乏能夠捕捉複雜 face image variation, 並且受梯度下降的初始值影響許多

2.  除了 2D sahpe, 3D model 的 face alignment 方法也正在發展

### Cascaded regression based
1.  Face alignment 是一個 regression 問題
    -   許多 regression-based 方法都被提出
    -   此方法嘗試學習一個 model, 將 image appearance map 到 target output

2.  這些方法的 performance 取決於 local descriptor 的 robustness
    -   Sun et al. [47] 提出 DCNN
        -   在每一個 level, 多個 network 的 output 會被混合並進行 landmark estimation
    
    -   Zhang et al. 提出 coarse-to-fine autoencoder networks 方法
        -   cascade 多個 stacked autoencoder networks (SANs)
        -   起始的幾個 SANs 預測每個 facial landmark 的位置
        -   隨後的 SANs refine landmarks
            -   以更高的解析度將 detect 到的 landmarks 周圍的 local features 作為 input

3.  另一方面, 由於不同的 dataset 提供不同的 facial landmark annotations
    -   在 Wild database 中的 300 個 Faces 被發展成可以公平比較各種 fiducial detection 方法的 benchmark
        -   包含了 12000 張 images, 以 68 個 landmarks 作標註
        -   包含 Labeled Face Parts, Helen, AFW, Ibug
        -   包含 600 張 test images
            -   300 indoor/ 300 outdoor

4.  除了用 2D transformation 來進行 face alignment
    -   Hassner [54] 提出利用 3D face model 來 frontalize face 的方法

    -   許多方法被發展出來以達成 face detection 上的 multitask learning (MTL)
        -   同時訓練 face detection task 和 correlated task 如 facial keypoint estimation

        -   MTL 可以幫助 network 產生 synergy(協同作用) 並學到更 robust 的 feature
            -   因為 network 得到了額外的 supervision
            -   例如在 keypoint 中得到的 eye center, nose tip 的資訊可以幫助 network 確定 face 的結構

## Face identification and verification
1.  在本章節，我們將會回顧 face identification 和 verification 的相關 works
    -   face identification 和 verification 有兩大要件
        -   robust face representation
        -   discriminative classification model 或是 discriminative similarity metric
            -   model 是 face identification
            -   metric 是 face verification

2.  傳統方法 [56]
    -   Local Binary Pattern (LBP)
    -   Fisher vector
    -   one-shot similarity kernel (OSS)
    -   Mahalanobis metric learning
    -   cosine  metric  learning
    -   large-margin  nearest  neighbor
    -   attribute-based  classifier
    -   joint Bayesian (JB)
    
### Robust feature learning for faces using deep learning
1.  face recognition system 中, 學到 invariant 且 disciminative 的 feature representations 是很重要的
    -   經過大量 data 訓練過的 DCNN 可以學到非常緊湊且有分辨力的 representation

2.  首先我們回顧一些使用 deep learning 的重要 face recognition work, 並且接著簡述現今 feature representation learning 的方法

3.  Huang et al. 提出利用 convolution deep belief network 來 learn face representation
    -   基於 local convolutional restricted Boltzmann machines

    -   首先用 unsupervised 的方式 learn representation
        -   沒有 label 的風景照 dataset
        -   透過 classification models transfer 學到的 representation 到 face identification/verification task
            -   SVM
            -   metric-learning approach (OSS)
        -   在 LFW dataset 上, 即使沒有大規模 annotated face data sets, 仍有成功效果

4.  Taigman et al. 在 DeepFace 方法中提出
    -   3D-model-based alignment along with DCNNs for face recognition

    -   利用 9層 deep neural network 並且使用多個 locally connected layers
        -   without weight sharing

5.  Sun et al. 提出 DeepID frameworks
    -   face verification
    -   ensemble 多個 shallower 和 smaller deep convolutional network 的結果

6.  Schroff et al. 提出 CNN-based 方法於 face recognition FaceNet
    -   直接最佳化 embedding 而不是 bottleneck layer

    -   使用 online triplet mining method 生成的 triplets
        -   roughly aligned matching
        -   non-matching

7.  Yang et al. [13]集成大量 annotated face data set CASIA-WebFace

8.  Parkhi et al. [17] 集成大量 annotated face data set VGGFace
    -   Train DCNN 
        -   based on VGGNet for object detection
        -   followed by triplet embedding for face verification

9.  在現今的 work 中, AbdAlmageed et al. [63] 將半身, 全身以及大頭貼分開分別訓練不同的 DCNN models

10. Masi et al.[64] 利用 3D morphable models 來 augment CASIA-WebFace data set
    -   合成的 face

11. Ding et. al. 提出 fuse facial landmark 周圍的 deep feature
    -   不同的 layer
    -   新的 triplet loss function

12. Wen et. al. [66]提出新的 loss function
    -   將每個 class 的 centroid 列入考慮並且以它為 softmax loss 的 regularization constraint

13. Liu et. al. [67]提出 angular loss, 調整 softmax loss
    -   可以用更少的 training data set 來達到最先進的 recognition performance

14. Ranjan et al. [68] 訓練 softmax loss regularized
    -   scaled L2-norm constraint

15. Yang et al. [70] 提出 neural aggregated network (NAN) 來動態調整多張 image aggregation 的權重

16. Bodla et al. [71] 提出 fusion network 來 combine 多個 DCNN 的 face representation

### Discriminative metric learning for faces
1.  從 data 學出一個 classfier 或 similarity measure 是 face recognition system 成功的關鍵
    -   face images
    -   face pairs

2.  Hu et al.[72] 學出 discriminative metric with deep neural network framework

3.  Schroff et al.[62] 和 Parkhi et al.[17] 將 DCNN 的 parameters 基於 triplet loss 做最佳化
    -   embed DCNN features 到 discriminative subspace

4.  Song et al. [74] 提出完全利用 training data in batch
    -   考慮 samples 中所有點對的距離

5.  Yang et al. [75] 提出在 recurrent framework 中學出 deep representations 以及 image clusters
    -   unsupervised learned representation 在 face recognition 和 digit classification 中都有良好表現

6.  Zhang et al. [76] 提出 cluster face images
    -   交替進行 deep representation adaption 和 clustering

7.  Trigeorgis et al [77] 提出 deep semisupervised 非負矩陣的 factorization 以學到 hidden representations
    -   使他們能夠依據 pose, emotion, identity 等未知 attribute 解譯出 clustering

8.  Lin et al. 提出 unsupervised clustering 演算法
    -   顯示出 samples 之間的 neighborhood structure
    -   進行 domain adaption

### Implementation
1.  Face recognition 可以分成兩個 task
    -   face verification
        -   給定一對 face image, 預測是否為同一人
    -   face identification
        -   將 image 與 database 作 matching 判斷是誰

2.  兩種 task 取得 discriminative and robust feature 都很重要
    -   face verification
        -   face 會先由 face detector 定位並且用 detect 到的 fiducial points 以 similarity tranform normalize 到 canonical coordinate

        -   之後每張 face images 通過 DCNN 來取得 feature representation

        -   feature representation 可用 similarity metric 計算出 score measure
            -   L2 distance
            -   cosine similarity (angular space)
        
        -   亦可 fuse 多個 feature 或是 similarity score

    -   face identification
        -   image 會經過 DCNNs 並且將 feature 儲存在 database 中
        -   新的 face image 輸入時，會先計算出 feature representation, 再與 database 中的 feature 計算 similarity

### Training data sets for face recognition
1.  MS-Celeb-1M 是目前最大的 public face recognition dta set
    -   包含超過 10 million labeled face images

2.  CelebA data set [80]
    -   202599 images of 10000 subjects
    -   40 face attributes
    -   5 keypoints

3.  CASIA-WebFace [13]
    -   494414 face images
    -   10575 subjects

4.  VGGFace [17]
    -   2.6 million face images
    -   2600 subjects

5.  MegaFace [14] [15]
    -   用以測試 face recognition algorithm 的 robustness
    -   open-set

6.  LFW [81]
    -   13233 face images
    -   5749 subjects 從 Internet 取得, 1680 subjects

7.  IJB-A data set [69]
    -   500 subjects
    -   5397 images
    -   2042 video 分成 20412 frames

8.  YTF [82]
    -   3425 videos
    -   1595 dubjects

9.  PaSC [83] dataset
    -   2802 video
    -   293 subjects

10. Celebrities in Frontal-Profile(CFP)
    -   7000 images
    -   500 subjects

11. UMDFaces [85]
    -   367888 still images
    -   8277 subjects

12. UMDFace Video [35]
    -   22075 video
    -   3107 subjects

13. still image 訓練出的 model 是否能用在 video 上
    -   fuse still image 和 video frame 更好
    -   smaller model
        -   wider dataset > deeper dataset
    
    -   deeper models
        -   wider dataset > deeper

## Performance summary
1.  總結現今 face identification 和 verification 應用於 LFW [81]和 IJB-A [69] dataset

### LFW datasets
1.  LFW dataset 包含 13233 face images, 5749 subjects
    -   其中 1680 subject 有兩張以上 image

2.  標準 facr verifiaction
    -   3000 張 positive pairs 和 3000 張 negative pairs
    -   分成十個 disjoint set, 每個 set 有 300 張 positive 300 張 negative, 作 cross validation

### IJB-A benchmark a data set
1.  IJB-A data set 包含 500 個 subject
    -   5397 張 images
    -   2042 個 video 分成 20412 frames

2.  face verification protocol
    -   平均一個 split 有 1756 pos 和 9992 neg

3.  face identifiction protocol 也有 10 個 split
    -   每個 split 大約有 112 gallery template 和 1763 probe templates
        -   1187 genuien 的 probe templates
        -   576 impostor probe templates
    
    -   training set 包含 333 subjects, test set 包含 167 subjects 且沒有重複

4.  IJB-A 將 image/video frames 分成 gallery 和 probe set
    -   包含多個 templates
    -   每個 template 包含 sample 出來 的 image 和 frame

5.  IJB-A 包含 extreme pose, illumination, expression 的 images
    -   LFW 和 YTF 則只包含 Viola Jones 的 face detector

## Facial attributes
1.  從一個 single face, 我們能夠 identify 出 facial attributes
    -   gender, expression, age, skin tone
    -   這些 attribute 可用以 image retrival, emotion detection, mobile security

2.  Kumar [56] 提出 image descriptor 的概念來描述 attributes
    -   65 種 binary attributes 來描述 face image

3.  Berg [56] 利用 classifiers 來生成 face verification classifier 使用的 feature

4.  每個人都以和別人的相似度來描述
    -   自動生成一系列的 attribute 而不是用 hand-label 的方式

5.  現今 DCNN 已被用以進行 attribute classification
    -   Pose Aligned Networks for Deep Attributes (PANDA)
        -   結合 part-based models 與 Deep learning 來訓練 pose-normalized DCNNs 以成為 attribute classifier [96]

    -   Liu [97] 使用兩個 DCNNs
        -   一個用以 face detection, 另一個用以 attribute recognition

6.  各個 attribute 之間其實存在著相關性
    -   [99] 利用 attribute 之間的 correlation 來提升 image ranking 和 retrieval
        -   以 classifier 的 output 計算 correlation
    
    -   Hand [100] 將 40 個 attribute 的關係以 network learn 出來, 不只是 attribute 的 pair

7.  Ranjan et al. [2] 訓練加上 MTL 的 single deep network
    -   同時可達到 face detection, facial landmark detection, face recognition, 3D head-pose etimation, gender classification, age estimation, smile detection

8.  Gunther et al. [101] 提出 Alignment-Free Facial Attribute Classification Technique (AFFACT)
    -   alignment-free facial attribute classification
    -   利用 data augmentation 來達到不需要於 bounding box 上作 alignment 下分類 facial attributes

## MTL for facial analysis
1.  MTL 首先由 Caruana [104] 進行分析
2.  Zhu [55] 將其用在 face detection, landmark localization 和 head-pose estimation 上
3.  JointCascade [105] 將 landmarks localization task 結合到 training 上

4.  在 deep-learning 出現前, MTL 被限制在某些 task 上，因為不同 task 需要的 representation 不同
    -   face detection 通常使用 HOG
    -   face recognition 通常使用 LBPs
    -   但現在 DCNN feature 取代了 handcrafted features, 因此通常可以訓練單一個 DCNN 來完成 multiple tasks

5.  當人類在看 face in image 時, 可以 detect 到 face 在哪裡以及判斷各個 attribute。然而 machine 通常會為各個 task 分別設計獨立的演算法
    -   但實際上可以讓所有 tasks 共用 features 並且呈現出 tasks 之間的關係
    -   MTL 可視為一種對 DCNN 的 regularization, 讓 network 能對所有 task 都通用
    -   HyperFace [10], Tasks-Constrained Deep Convolution Network (TCDCN) [107] 皆為 DCNN 加上 MTL
    
## Open issues
1.  Face detection
    -   face detection 的挑戰為多變的
        -   pose, illumination, view point, occlusions

2.  Fiducial detection
    -   DCNN 可以取出臉部較抽象的資訊，但目前尚未明瞭哪些 layers 確切對應到的 local features

3.  Face identification / verification
    -   在記憶體的限制上，如何選擇較具備 information 的 pairs/triplets 來作 training