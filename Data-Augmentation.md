### Shorten, C., Khoshgoftaar, T.M. A survey on Image Data Augmentation for Deep Learning. J Big Data6, 60 (2019).

1.  Data Augmentation
    -   image Augmenation 演算法包含
        -   geometric transformations
        -   color space augmentation
        -   kernel filters
        -   mixing images
        -   random erasing
        -   feature space augmentation
        -   adversarial training
        -   generative adversarial networks
        -   neural style transfer
        -   meta-learning

2.  其他 Data Augmentation 的特徵
    -   test-time augmentation
    -   resolution impact
    -   final dataset size
    -   curriculum learning

3.  本survey將會呈現現有的Method for
    -   Data Augmentation 
    -   Promising Development
    -   meta-level decision

-   Overfitting 就是 network學出了一個function。此function有很高的變異數，只對training data好
-   讀者將能夠了解Data Augmentation是如何提升model的效能。擴展有限的dataset以取得big data的效果

### Introduction
1.  Deep Learning model 已經對於"識別"(discriminative)有了一定程度的能力，歸功於
    -   deep network architecture
    -   powerful 運算
    -   能夠 big data

2.  Deep Neural network 成功地運用在 Computer Vision 上 (CNN)
    -   image classification
    -   object detection
    -   image segmentation
    稀疏連結的kernel、參數來保留image的效果

3.  Convolution Layer會
    -   downsample 影像的 spatial resolution
    -   增加 feature map 的深度

-   CNN的成功讓Deep neural network 在 computer network 上的運用受到重視

4.  提升 generalization ability 是一大挑戰
    -   generalization ability 意指 model 在看從沒看過的data以及已經看過的data的表現差異
    -   差勁的generalization ability會帶來overfitting

### Background
### Image Data Augmentation Techniques
1.  Data Augmentation based on basic image manipulations
    <Geometric transformation>
        -   Geometric transformation會有safety上的問題。例如翻轉變換可能會讓6和9出現混淆
    1.  Flipping
        -   對水平軸翻轉比垂直軸更常見，是最簡單的作法且在CIFAR-10與ImageNet上有很好的效果
        -   但在文字辨識的dataset上(如同MNIST, SVHN)，不是label-preserving的transformation
    2.  Color Space
        -   影像data通常由RGB三個channel構成
        -   最簡單的作法為將某兩個channel歸零，留下單一顏色
        -   也可以調亮度
        -   也可以利用color histogram來更動image的顏色分佈
    3.  Cropping
        -   如果image data中有各種長寬不一的照片混合在一起，可以將image中間的patch擷取出來就好
        -   會改變image的大小
        -   可能不能label-preserving
    4.  Rotation
        -   旋轉照片在角度小的時候很有用，但在角度大的時候就未必能夠label-preserving
        -   小角度旋轉對MNIST等手寫資料集效果不錯
    5.  Translation
        -   對image data做平移可以避免掉 position bias
        -   如果training data多半是正中央(像是人臉辨識)，則可能導致model只能處理在中央的test data
        -   平移之後，留下來的部份可以補0, 255或是random/gaussian noise
    6.  Noise injection
        -   將一個每個值由gaussian取樣出來的matrix加到image上
        -   經過Moreno-Barea測試，可以讓CNN得到更robust的feature
    -   disadvantages
        -   需要多餘的空間與運算資源、運算時間
        -   某些作法需要人工檢驗是否label-preserve
        -   實際場景上，使training data和testing data的分佈產生差異的部份可能不只是geometric上

    <Color space transformation>
    1.  過暗或過亮的image，可以在整張image上統一加減一個值
    2.  image的直方圖可以用來apply一個filter來操作color space的特徵
    3.  RGB換成grayscale可以達到更快的運算速度，但是會降低準確度
        -   ImageNet 在 PASCAL、VOC dataset上有3%左右的準確度降低
    
    4.  Jurio et al.有針對各個color space上Image Segmentation的研究
        -   RGB, YUV, CMY, HSV

    -   disadvantages
        -   training time, increased memory, transformation costs的增加
        -   可能不是label-preserve
            -   如果降低pixel value，將可能看不見某些object
        -   在Image Sentiment Analysis(圖片情感分析)上，如果color space改變到blood，model可能會得到較差的表現

    <Geometric versus photometric transformations>
    1.  Taylor and Nitschke做了geometric和color space兩個方法的比較
        -   使用Caltech101 dataset的8421張256x256的照片，計算4-fold cross validation
        -   baseline 為 48，所有作法都有提升，其中以Cropping得到最好的進步
        
    <Kernel Filters>
    1.  Kernel Filter 在 sharpen和blur image上很常使用
    2.  blur image可以讓model對motion blur有更好的抵抗力
    3.  sharpen image則可以讓model對object得到更多detail
    `Sharpening and blurring`
    -   disadvantages
        -   kernel filter在Data Augmentation中比較少見
        -   與CNN較相似，可能可以實作在convolution layer中而不是在dataset中。

    <Mixing Images>
    1.  將多張Images做平均取得新的Images，直覺來說應該無法達到Data Augmentation的效果
    2.  Ionue做了實驗，將一對sample出來的data作平均，得到新的data並且使用同樣的label
        -   for each image in training data
                sample one image in training data
                averaging them
        
        -   CIFAR-10 dataset中，Ionue成功將錯誤率降低(8.22%-> 6.93%)
        -   甚至可以降低data 使用量
    
    3.  Summers and Dinneen更進一步研究用non-linear的方式來combine兩張image以取得新的data
        -   皆比baseline更好
    
    4.  Liang et  al使用GANs來取得mixed images。
        -   若將mixed images加入training dataset中，可以降低訓練時間並且增加GANs-samples的diversity
    
    5.  Takahashi and Matsubara實驗將各個images切割並隨機重組，也是相當不直覺的作法

    -   disadvantages
        -   違反人的直覺，難以解釋
        -   勉強解釋:增加的data量可以得到更robust的low-level特徵(lines、edges)
    -   Transfer learning 以及 pretraining是其他可用來學到low-level特徵的作法
        -   常用來和Mixing Images做比較

    <Random Erasing>
    1.  Zhong et al.從dropout regularization得到random erasing的靈感。
        -   隨機從image中拿掉一塊
    2.  Random Erasing專門設計用來對付occlusion(部份image不清楚)
        -   防止model去 overfit 特定visual feature
        -   讓model注意整張image而不是subset of image
    
    3.  Random Erasing是隨機選一塊nxm大小的patch，將其mask成0,255或隨機數
        -   在CIFAR-10 dataset中可將錯誤率從5.17%降到4.31%
    
    4.  逼迫model學習更descriptive的feature
        -   和color filter, horizontal flipping同為top augmentation technique
    
    5.  或許可以用不同形狀的erase
        -   各種erase configuration
    
    -   disadvantages
        -   不總是label-preserving(8可能被擦掉變成6)
        -   Stanford Cars dataset可能會產生品牌無法辨識的情形

    <A note on combining augmentation>
    1.  若是training data的量非常少，混合使用這些雖可得到極為膨脹的data量，但很可能會overfitting
    2.  需要考慮augmented space 的 search algorithm (Design Consideration)

2.  Data Augmentation based on Deep Learning
    <Feature Space augmentation>
    1.  Neural Network 可以用來將高維度的data 映射到低維度的representation
        -   map image to binary class/ nx1 vector
        -   可以從feature層開始增加data數

    2.  中間層也可以從model分出來
        -   Konno and Iwazume 發現 CIFAR-100 在各層分割出來之後可以有從66%到73%的進步
    
    3.  CNN 中的 High-level Layer 被認為是 feature space
        -   Data Augmentation 多了許多 vector operations
    
    4.  SMOTE 在class imbalance 上相當常用
        -   聚集k nearest neighbors來取得新的data instances
    
    5.  autoencoder即是用來作feature augmentation
        -   前半段是encoder，可以用來將image映射到低維度的vector representation
        -   後半段是decoder，可以用來將vector representation恢復到原本的images
        -   encode出來的representation即可用來做feature space的augmentation
        -   DeVries and Taylor 利用 3-nearest-neighbor來外推出新的data，並且和在input space作外推的結果做比較
    
    6.  feature space augmentation的作法有
        -   用auto encoder
        -   去除CNN的output space，留下中間的representation
            -   這些representation也能用來train任何model。如 Naive Bayes, SVM, DNN
    -   disadvantage
        -   要解釋vector data很困難
        -   auto encoder完全複製encode層，很花費時間
        -   Wong et  al. 發現data space的表現會比feature space好
    <Adversarial training>
    1.  
    <GAN-based Data Augmentation>
    1.  Generative modeling 指的是人工產生保留原dataset特徵的data instance
        -   Bowles et  al.指 GANs 是解鎖dataset中額外資訊的方式
        -   GAN 不是唯一的 generative model，但在運算速度和效果都相當出色
    <Neural Style Transfer>
    1.  Neural Style Transfer 操作 CNN 從image中抓出來的representation
        -   在保有原本內容的情況下，改變image的style
    
    2.  Fast Style Transfer將原本per-pixel的loss轉化成perceptual loss，並使用feed-forward network來stylize image

    3.  Ulyanov et al.發現將batch normalization換成instance normalization有更好的表現

    4.  Neural Style Transfer 在 data augmentation上的實作可以透過
        -   選擇一系列的k個styles，作用在training set上的所有images
        -   Style Augmentation可以避免style bias
    
    5.  Tobin et al.利用不同在模擬training中運用不同styles，成功做到準確度達1.5cm的object localization
        -   實驗中randomize物體的位置和texture
        -   像是光線、光線數、texture，以及背景的noise
    
    6.  在電腦視覺中使用simulated data目前被重點研究
        -   Richter et al. 利用open-world games來產生semantic segmentation datasets
            -   標註花費成本相當高
            -   CamVid dataset 每張照片需花60分鐘做標註
            -   Cityscapes dataset 每張照片需花90分鐘作標註
    -   disadvantages
        -   需要選擇將image transfer進去的style
        -   如果styles set太小，可能會產生biases

    <Meta Learning Data Augmentations>
    1.  Meta Learning 一般是指利用Neural Network來optimize Neural Network
        -   random search
        -   evolutionary strategy
        -   本篇僅先以neural-network, gradient-based為主
    2.  歷史演進
        -   feature engineering (SIFT, HOG)
        -   architecture design (AlexNet, VGGNet, Inception-V3)
        -   下一個可能為 meta-architecture design

    3.  Neural augmentation
        -   Neural Style Transfer需要
            -   weight of style
            -   content loss
        -   Neural Augmentation 會 meta-learn 一個 Neural Style Transfer strategy
            -   從同一個class中隨機取出兩張image
            -   一張丟入CNN，並將得到的結果與另一張放到Neural Style Transfer作transformed
            -   此處的CycleGAN使用Neural Style Transfer
            -   此兩張圖片再放入分類用的model以計算loss，並用以update Neural Augmentation Net

    4.  Smart Augmentation
        -   Smart Augmentation類似Neural Augmentation，但是combine兩張image的部份使用預先準備好的CNN而不是Neural Style Transfer algorithm

        -   採用兩個networks A、B，A為augmentation network，輸入多張image並且輸出一張新的image給B做訓練
        -   B的loss將會backpropogate來update A

        -   類似於 mixed-examples 以及 SamplePairing，但是使用的是比取average更adaptive的CNN

    5.  AutoAugment
        -   AutoAugment 為一種 Reinforcement Learning algorithm
            -   以一系列的 geometric transformations 為 search space
    
    -   disadvantage
        -   meta-learning的概念較新，尚未被仔細測試
        -   meta-learning可能實作上較困難、耗時
            -   vanish gradients 等問題


    6.  Comparing Augmentations

### Design Considerations for Image Data Augmentation
1.  Feature space augmentation
2.  Test-time Augmentation
3.  Curriculum Learning
4.  Resolution Impact
5.  Final dataset size
6.  Alleviating class imbalance with Data Augmentation

### Discussion
### Future Work
### Conclusion