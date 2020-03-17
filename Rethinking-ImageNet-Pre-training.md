# Rethinking ImageNet Pre-training

## Abstract
1.  使用COCO dataset以及`隨機初始化`的standard model，可以得到不遜於ImageNet Pretrain過的結果
    -   使用更多Iteration使隨機初始化的model收斂

2.  利用`隨機初始化`在以下情況有更robust的結果
    -   十分之一的training data
    -   更deeper/wider的model
    -   multiple tasks/ metrics

3.  pretraining可以加速收斂，但是不一定能夠有regularization的效果，也不一定保證正確率
    -   50.9 AP 於 COCO dataset 上，沒有任何外部data，仍有和ImageNet Pretraining相當的正確率

## 1. Introduction
1.  現今Computer Vision採用大規模的data來對model作pretrain，再將model依照目標task作fine-tuning
    -   object detection, image segmentation, action recognition

2.  本篇paper提出
    -   object detection 和 instance segmentation 的準確度能夠用隨機初始化達到而不需pre-train
    -   甚至也能使用baseline system與hyper-parameters
    -   處理好以下情況，就未必需要pre-train
        -   適當使用normalization
        -   投入足夠久的時間訓練模型，以補償未經pre-train

3.  經過實驗過後的發現：
    -   ImageNet pre-training可以加快收斂、縮短訓練時間，但是從頭開始訓練仍然可以追上pre-train的結果
        -   大約是ImageNet pre-train加上fine-tuning的時間
        -   從頭學習 pre-train 時會學到的 low-level feature (edges, textures)
    
    -   ImageNet pre-training沒有自動給出更好的regularization
        -   當資料量少時，必須提供新的 hyper-parameter 來作 fine-tuning 以免 overfitting
        -   但從頭開始train就能夠用一樣的 hyper-parameter，不用額外加上 regularization
    
    -   如果目標 task 對 spatial 上的位置有較高的要求，ImageNet pre-training 可能沒有優勢

4.  結果與挑戰：
    -   pre-training 在 target data 較少以及沒有那麼多運算資源做重新訓練的情況較適合
    -   然而，在運算資源和 target data 充足的情形下，本研究建議可以進行重新訓練

## 2. Related Work
<Pre-training and fine-tuning>
1.  現今多數 object detect 都是以 pre-training and fine-tuning 的模式來進行
2.  近代研究則更將 data 量增加到 ImageNet 的5倍、300倍、3000倍
    -   但增加 data 的邊際效益卻逐步減少

<Detection from scratch>

1.  與眾多論文不同的是，本研究專注在了解 ImageNet pre-training 於 unspecialized architectures
    -   探討 with and without pre-training
    -   unspecialized 意指 model 在設計上並沒有考慮要從頭 train

## 3. Methodology
1.  希望在實驗中得到 `pre-training 可以不用` 這個結果
2.  實驗過程中，在 train from scratch 之後，於 model 上加的 modification 要盡可能少，只用以下兩個

### 3.1. Normalization
1.  Image Classifier 需要 normalization 來幫助最佳化
    -   Batch Normalization 為常見作法，但可能會讓 object detector 於從頭 training 造成困難
    -   兩種作法可以代替 BN
        -   Group Normalization (GN)
            -   GN 的運算與 batch dimension 無關
            -   GN 的準確度不受 batch size 影響
        -   Synchronized Batch Normalization (SyncBN)
            -   利用 multiple device 計算，可以避免掉 small batches
### 3.2. Convergence
1.  training from scratch 的收斂速度不可能和 pre-train 相同，這可能讓實驗者對 model的效果做出錯誤的結論
2.  pre-training model 已經學好 low-level 的 features，在 fine-tuning 的階段不需要重學
    -   training from scratch 則必須同時學 low-level 和 high-level semantics

3.  training from scratch 必須要有比 fine-tuned 更長的訓練時間，看過更多的 sample。sample有以下三種定義
    -   image 數
    -   instance 數
    -   pixel 數

4.  雖然從頭 train 會需要將近3倍於 fine-tuning 的 iterations，但如果把 pre-train 的 iteration 數也算入，事實上 training from scratch 都用更少的 iterations
    -   只有 pixel 數勉強相近

## 4. Experimental Settings
<Architecture.>

1.  Mask R-CNN, ResNet, FPN(Feature Pyramid Network)
2.  End-to-End, Region Proposal Networks
3.  GN/SyncBN (pre-train model 也會用 GN/SyncBN)

<Learning rate scheduling.>

1.  原先的 Mask R-CNN 會以 90k(schedule) 的 iterations 來做 fine-tune
2.  training from scratch 則可能會拉長時間(6 x schedule)

<Hyper-parameters.>

1.  多半 hyper-parameter 會依據 Detectron
    -   initial learning rate
    -   weight decay
    -   momentum

2.  no data augmentation
    -   只有 horizontal flipping

## 5. Results and Analysis
### 5.1. Training from scratch to match accuracy
1.  只使用 COCO data 時，training from scratching 可以追上 fine-tune 的正確率
2.  利用 COCO train2017 split (~118k images)當作訓練，val2017 (~5k images)來做validation
    -   以 Average Precision (AP) 衡量 object detection
    -   以 mask AP 衡量 instance segmentation

<Baselines with GN and SyncBN.>

1.  比較同一個 model 在 random initialization 和 fine-tuned with ImageNet pre-training 的效果
    -   5種不同的 schedules
    -   accuracy的飛躍是因為 learning rate 的減低

2.  一致的現象
    -   典型的 fine-tuning schedule 需要(2x) 來達到收斂
    -   training from scratch 可以追上 fine-tuning 於 (5x 或 6x) 的 schedule
    -   當收斂時，兩者並無正確率上的差異
<Multiple detection metrics.>

1.  用兩種衡量方式來比較 training from scratch 以及 pre-training
    -   box-level AP
    -   segmentation-level AP
    -   IoU threshold of 0.5 與 0.75
    -   在 AP_50 和 AP_75，training from scratch 皆表現的比 pre-training 好

<Enhanced baselines.>

1.  – Training-time scale augmentation:
    -   到目前為止，model 都沒有作 horizontal flipping 的 data augmentation
    -   現在加入一些 training time 的 augmentation
        -   images的短邊隨機sample出 [640,800] pixels 的 images
        -   使用augmentation會需要更多時間來收斂
            -   training from scratch 9X
            -   pre-train 6X
        -   高度的data augmentation可以用來解決data量少的問題
            -   pre-train比較沒有得到助益

2.  – Cascade R-CNN
    -   Cascade 注重 localization 準確度。追加兩個stage於標準的 two-stage Faster RCNN
    -   實作它的 Mask R-CNN 版本，本篇論文於最後一個stage加上 mask head

3.  – Test-time augmentation:
    -   結合從 multiple scaling transformation 得到的 prediction 來做 test-time augmentation

<Large models trained from scratch.>

1.  以 ResNeXt-152 8×32d 加上 Group Normalization 為基底，從頭訓練出更大的 Mask R-CNN
    -   此 model 有4倍於 R101 的 FLOPs
    -   雖然 model 規模大，但沒有明顯的　overfitting
    -   training from scratch 仍然有比 pre-training 更好的結果

<vs. previous from-scratch results.>

1.  過往的結果並沒有呈獻出 未經pre-train的model 能夠表現出和 經過pre-train的model 有相當的表現

<Keypoint detection.>

1.  Key point detection 是更加需要 spatial localization 的 task
    -   training from scratching 更快追上 pre-training

<Models without BN/GN — VGG nets.>

1.  即便加上 pre-training，VGG的收斂速度依然相當慢
2.  大約在 11x schedule，from scratching 可以追上 pre-training

### 5.2. Training from scratch with less data

1.  model training from scratching 使用十分之一的data，可以得到和pre-training相當的結果
    -   35k COCO training images.
    -   10k COCO training images.
    -   Breakdown regime: 1k COCO training images.
        -   大約是全COCO dataset的百分之一
        -   training loss 可以追上 pre-training，但 validation loss 沒有辦法
        -   嚴重缺乏data而導致的overfitting

    -   Breakdown regime: PASCAL VOC.
        -   在 trainval2007+train2012 混合dataset上訓練，並在 val2012 上測試
        -   from scratching 沒能追上 pre-training，很可能是因為
            -   VOC images 每張大約只有 2.3 個 instances (COCO 有約7個)
            -   VOC 只有 20 個 categories (COCO 有約80種)
            -   缺乏 instances 和 categories 相當於缺乏 training data