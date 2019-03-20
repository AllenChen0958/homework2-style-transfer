
ISA525700 Computer Vision for Visual Effects<br/>Assignment 2: MUNIT
===



## Abstract
現今3C資訊產品市場規模龐大，消費者對於電腦視覺品質要求，亦日益升高。本文所探討MUNIT model具有一對多影像輸出轉換效果，對於其轉換後影像品質與多樣性等層面效果如何?這是在本文與實驗中欲深入探討!

Keyword: 電腦視覺, MUNIT, 影像品質, 多樣性

## Introduction

- 現今全球電腦、平板與智慧型手機等3C資訊產品普及，消費者對電腦視覺的品質與多樣性，亦日益增高。在手機方面，根據[IDC 2018年第3季調查報告](https://3c.ltn.com.tw/news/35026)指出，全球智慧型手機總出貨量約3.5億支，市場規模龐大。

- 因UNIT model僅能一對一影像轉換，因此Xun Huang等四位學者在2018年Cornell University與NVIDIA的產學合作計劃中，發表提出MUNIT model，成功突破此難題，挑戰一對多影像生成，成功轉出多張影像。而本文正探討MUNIT的影像品質與多樣性等電腦視覺效果，與其它ML Model比較是否具有提供消費者最佳品質。

- 讓我們觀看下面關於MUNIT實驗成果影片。

<iframe width="727" height="409" src="https://www.youtube.com/embed/ab64TWzWn40" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Method: UNIT
非監督式影像對影像轉換法(Unsupervised Image-to-Image Translation, UNIT)，可以對兩個dataset進行Training，如老虎與獅子兩種照片，可以深度學習到眼睛、鼻子、嘴等等五官之共同特徵。

## Method: Bicycle GAN
Bicycle GAN為監督式學習(supervised learning)，經過pair instances即可訓練出共同特徵，再透過亂數(Random)改變輸出，可以生成一對多。

## Method: MUNIT
- 本文探討多模態非監督式影像對影像轉換法(Multimodal Unsupervised Image-to-Image Translation, MUNIT)。此模型主要透過內容碼(content code)與樣式碼(style code)學習圖片生成。

- 多模態(Multimodal)概念為生成一對多圖片風格轉換(style transfer)，例如將老虎的圖片轉成多張獅子圖片。Multimodal有隨機模型(stochastic model)與確定型模型(deterministic model)兩種。隨機模型(stochastic model)是輸入一張圖片可輸出多張隨機不同圖片。確定型模型(deterministic model)是一對一固定輸出圖片。

- 非監督式(Unsupervised)為將兩個不相同dataset一起訓練，例如老虎與狗照片，學習牠們五官特徵。

- 內容碼(content code)是學習共同特徵，例如將老虎的臉部五官特徵學習出，變成多張獅子照片。風格碼(style code)功能為將原圖轉換多張不同風格的圖片，例如秋天原圖轉成春季影像。如下圖所示，X1為輸入圖片，X2輸出圖片，C為content code，S1與S2為style code，X2是依據S2亂數產生多張不同圖片。

![](https://i.imgur.com/nb7pHx0.png)
 
- 如果想將一張影像風格套用在另一張影像的內容，我們可以從深度學習網路中擷取內容與風格兩種特徵，再將兩者進行重建，並採用損失函數與最佳演算法找出最佳影像輸出。


## Loss Function
![](https://i.imgur.com/QPyuVNK.png)

上述loss function 可拆成兩部份，GAN本身的loss($\mathcal{L}^{x_i}_{GAN}$) 加上 reconstruction loss($\lambda_k(\mathcal{L}^{k_1}_{recon}+\mathcal{L}^{k_2}_{recon})$)

- GAN loss($\mathcal{L}^{x_i}_{GAN}$): 該部份的loss即是傳統GAN的loss，主要目地在於讓Discriminator能有分出Generator產生的圖片和真正的圖片$x_i$，同時讓Generator產生的圖片越來越逼真而能夠騙過Discriminator

- reconstruction loss:
共可分為以下三種形式
    - ![](https://i.imgur.com/079YaZm.png)
    > 第一種是讓content encoder，以及style encoder將原圖分別enode後，將這些encode後的資訊拿來給GAN產生圖片，和原圖相同
    - ![](https://i.imgur.com/3PBZe5e.png)
    > 第二種是希望content code能在decode和encode過程中保留(EX: 狗轉貓後，貓和狗的五官(content code)希望都保留下來且是相同的)
    - ![](https://i.imgur.com/amew9hy.png)
    > 第二種是希望style code能在decode和encode過程中保留(狗轉某特定貓後，能夠用decoder得知該特定貓的品種(style code))


## Training Process
如下圖所示，MUNIT影像訓練，一個cycle單位為一萬。

![](https://i.imgur.com/K0fJtRy.jpg)


## Inference Munit in personal image

### 1.Pick 3 different paintings
---
訓練完成該畫風的模型後，選其中三張畫作做為style參考，選三張真實照片(一張自己提供兩張由datasets隨機選出)分別就不同style進行轉換，其中公園的真實照片轉成style2的畫作最為成功!判斷是因為其情境十分相似，但普遍效果不佳，所以決定直接產生style code來對照片進行轉換。
#### a.Photo2Monet
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/NuFTDkZ.jpg)|![](https://i.imgur.com/lVyF84M.jpg)|![](https://i.imgur.com/VqONRxd.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/Nl6BLum.jpg)|![](https://i.imgur.com/MuX5MPi.jpg)|![](https://i.imgur.com/JqIo6wT.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/Ox2UAMS.jpg)|![](https://i.imgur.com/JyCTh4H.jpg)|![](https://i.imgur.com/htA5CZa.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/9BYzAiP.jpg)|![](https://i.imgur.com/WfsSU6A.jpg)|![](https://i.imgur.com/98YEb3R.jpg)|


#### b.Photo2Vangogh
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/effAhcg.jpg)|![](https://i.imgur.com/GJJVWgi.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/oIEitHE.jpg)|![](https://i.imgur.com/IpX6YcI.jpg)|![](https://i.imgur.com/2oCwpD2.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/R79DEUA.jpg)|![](https://i.imgur.com/TVu67O3.jpg)|![](https://i.imgur.com/hLnxhrY.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/L6QT7OJ.jpg)|![](https://i.imgur.com/WIiRck6.jpg)|![](https://i.imgur.com/rUkSHs4.jpg)|

#### c.Photo2Cezanne 
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/KdgdtLu.jpg)|![](https://i.imgur.com/MYFTUBs.jpg)|![](https://i.imgur.com/YQ8HW1G.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/IcT2iE6.jpg)|![](https://i.imgur.com/Dqze08V.jpg)|![](https://i.imgur.com/QEEpV67.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/dBGmUTn.jpg)|![](https://i.imgur.com/E6iEhqh.jpg)|![](https://i.imgur.com/V3aACNP.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/08ecHL8.jpg)|![](https://i.imgur.com/ZhRmWZE.jpg)|![](https://i.imgur.com/zRuMIxI.jpg)|
### 2.Linearity produce style code
---
自己產生style code 進行轉換我們使用了兩種技巧，一種是技巧A,隨機產生兩個style code並以兩者變化量除16做為單位向量，此向量有8個維度，因為儲存style_dim在訓練時設定為8，技巧B則是一次只動style code的其中一個維度，範圍從-7到0.5，遞增0.5，其他維度設為0。
#### 技巧A

向量調整style code
![](https://i.imgur.com/1i8H6P0.jpg)

|![](https://i.imgur.com/D1KCYtB.gif)|![](https://i.imgur.com/94PG8rX.gif)| ![](https://i.imgur.com/GbdkQii.gif)|
|:--:|:--:|:--:|
|cezanne|monet|vangogh|
因為是用兩個隨機的style code產生單位向量，所以每個維度變化尺度不同，畫風線性變化上不一定很明顯，不過產生的圖片在風格上很接近畫作相較於直接抽取畫作的style code。
#### 技巧Ｂ
之所以維度調整範圍由-7到0.5是因為幾乎每個維度在這個範圍內都有明顯的線性變化。




|dimension欄位|Monet|Vangogh|Cezanne|
|:--:|:--:|:--:|:--:|
|1|![](https://i.imgur.com/2yP0Owf.gif)|![](https://i.imgur.com/6eoWLaM.gif)|![](https://i.imgur.com/tqPyTCe.gif)|
|2|![](https://i.imgur.com/nUwQBic.gif)|![](https://i.imgur.com/juQNtuj.gif)|![](https://i.imgur.com/yL6ncCT.gif)|
|3|![](https://i.imgur.com/c73cuMO.gif)|![](https://i.imgur.com/6cnKgCc.gif)|![](https://i.imgur.com/a8mtXN1.gif)|
|4|![](https://i.imgur.com/UXzXfiP.gif)|![](https://i.imgur.com/LeEbio0.gif)|![](https://i.imgur.com/stiCU8L.gif)|
|5|![](https://i.imgur.com/zOwSJNN.gif)|![](https://i.imgur.com/DWOTKNK.gif)|![](https://i.imgur.com/TKSj19F.gif)|
|6|![](https://i.imgur.com/83xIPq3.gif)|![](https://i.imgur.com/yJElZuC.gif)|![](https://i.imgur.com/Lj2DKQJ.gif)|
|7|![](https://i.imgur.com/sHk22jR.gif)|![](https://i.imgur.com/Xm2TyT5.gif)|![](https://i.imgur.com/bG5hsVs.gif)|
|8|![](https://i.imgur.com/YEo1sPN.gif)|![](https://i.imgur.com/QbGQqz4.gif)|![](https://i.imgur.com/6vfDPCi.gif)|

主觀上，雖然相同dimension不同畫家產生線性變化不同，調整style code時，其中一個畫家產生的圖片具有另外兩個畫家畫作的特徵，有可能是因為三者皆為印象派畫風。

### 3. Mix datasets to enhance the variance of style code
---
經過前一部分的實驗，我們猜測某種畫家的畫風，會被encode在某個style dimension，故該部分我們嘗試將三種不同派別畫家(or畫風)的畫混合起來做training，看是否能學到更多元的style code。

其中所選擇的三位畫家(畫風)是：浮世繪(ukiyoe)、莫內(Monet)、保羅尚賽(cezan)的畫。

這裡有個背景知識：浮世繪的油畫風格，有深刻影響到印象派(莫內)發展，而後更是受立體主義影響，從印象派演變至後印象派(保羅尚賽)。因此我們假設這三種畫風其實存在content code(演變過程保留的feature)，而其有別於其他畫派的特色可用不同的style code描述。

其中以下的測試資料當中，梵谷(Vango·新印象派)的畫(如下圖b.)是沒有訓練過的，但仍然能夠產生類似的畫風。且相較於直接用梵谷的畫作train出來的效果要好。
###### 備註：混合三種畫的model train到110000 iteration，純粹用梵谷的畫train的model則是到120000 iteration

#### a.Photo2Monet
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/NuFTDkZ.jpg)|![](https://i.imgur.com/lVyF84M.jpg)|![](https://i.imgur.com/VqONRxd.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/4olP9ia.jpg)|![](https://i.imgur.com/te2PLHK.jpg)|![](https://i.imgur.com/6Ggznrn.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/9hCBO18.jpg)|![](https://i.imgur.com/HHRsorN.jpg)|![](https://i.imgur.com/xImDNmX.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/lKl9bxj.jpg)|![](https://i.imgur.com/8yQWqmp.jpg)|![](https://i.imgur.com/UpjW2HP.jpg)|


#### b.Photo2Vangogh
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/effAhcg.jpg)|![](https://i.imgur.com/GJJVWgi.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/rzqyM5N.jpg)|![](https://i.imgur.com/4vFWXyY.jpg)|![](https://i.imgur.com/LbMhpxW.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/qCmCXgr.jpg)|![](https://i.imgur.com/ItUGOuX.jpg)|![](https://i.imgur.com/CqktXTA.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/eBmzQ1B.jpg)|![](https://i.imgur.com/yLkzMyB.jpg)|![](https://i.imgur.com/7PDx1Cx.jpg)|

#### c.Photo2Cezanne 
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/KdgdtLu.jpg)|![](https://i.imgur.com/MYFTUBs.jpg)|![](https://i.imgur.com/YQ8HW1G.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/a1FNcI6.jpg)|![](https://i.imgur.com/mAIJef2.jpg)|![](https://i.imgur.com/tpfILqy.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/OkjezqA.jpg)|![](https://i.imgur.com/vuGsLKC.jpg)|![](https://i.imgur.com/PeMTz4s.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/07kH3NT.jpg)|![](https://i.imgur.com/LRCnx5I.jpg)|![](https://i.imgur.com/JaB8iFU.jpg)|



## Other Methods

### [A Neural Algorithm of Artistic Style](https://ithelp.ithome.com.tw/articles/10192738)

#### Example
為方便閱讀比較，該方法的測資上，選擇和上方MUNIT三張畫作作為style參考，並依照MUNIT的三張真實照片分別就不同style進行轉換，該方法會隨其pretrained vgg-model的好壞
#### a.Photo2Monet
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/NuFTDkZ.jpg)|![](https://i.imgur.com/lVyF84M.jpg)|![](https://i.imgur.com/VqONRxd.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/QlV20SQ.jpg)|![](https://i.imgur.com/T7Bq1GP.jpg)|![](https://i.imgur.com/If3sje2.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/hbUuDCs.jpg)|![](https://i.imgur.com/1fVXnRC.jpg)|![](https://i.imgur.com/zeUqxnU.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/gFlea6o.jpg)|![](https://i.imgur.com/dFUFiDa.jpg)|![](https://i.imgur.com/cCPkNXT.jpg)|


#### b.Photo2Vangogh
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/effAhcg.jpg)|![](https://i.imgur.com/GJJVWgi.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/SesPf3s.jpg)|![](https://i.imgur.com/nVRGcgY.jpg)|![](https://i.imgur.com/5ConY02.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/edJEfHT.jpg)|![](https://i.imgur.com/24KxhfV.jpg)|![](https://i.imgur.com/ENMI9Z7.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/2BLWcPQ.jpg)|![](https://i.imgur.com/ziAuKE1.jpg)|![](https://i.imgur.com/ki6YZgR.jpg)|

#### c.Photo2Cezanne 
|![](https://i.imgur.com/RuFjck5.png)|![](https://i.imgur.com/KdgdtLu.jpg)|![](https://i.imgur.com/MYFTUBs.jpg)|![](https://i.imgur.com/YQ8HW1G.jpg)|
|:-----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/erEpnYK.jpg)|![](https://i.imgur.com/Dqze08V.jpg)|![](https://i.imgur.com/1Sb0hXJ.jpg)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/5qKREXR.jpg)|![](https://i.imgur.com/leMdjAi.jpg)|![](https://i.imgur.com/DQOQNCW.jpg)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/vc878Be.jpg)|![](https://i.imgur.com/voLlx5q.jpg)|![](https://i.imgur.com/tI9AMlh.jpg)|

### Comparisons

#### Form contains differnt methods (photo2Vangogh)
| _content_  | _style_  |  munit |Cycle GAN|fast transfer | Neural Algorith|Diverse img transfer| 
|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|
|![](https://i.imgur.com/gsesz2Z.png)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/oIEitHE.jpg)|![](https://i.imgur.com/p0ZSDXs.png)|![](https://i.imgur.com/DmjDwm0.png)|![](https://i.imgur.com/SesPf3s.jpg)|![](https://i.imgur.com/80zeWHn.png)|
|![](https://i.imgur.com/7z8Ie5q.jpg)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/R79DEUA.jpg)|![](https://i.imgur.com/ipJIWGZ.png)|![](https://i.imgur.com/cnjugPS.png)|![](https://i.imgur.com/SesPf3s.png)|![](https://i.imgur.com/5zNg0Zu.png)|
|![](https://i.imgur.com/LB1UoYF.jpg)|![](https://i.imgur.com/4KVTN7v.jpg)|![](https://i.imgur.com/L6QT7OJ.jpg)|![](https://i.imgur.com/amORqBD.png)|![](https://i.imgur.com/iJndGNW.png)|![](https://i.imgur.com/2BLWcPQ.png)|![](https://i.imgur.com/JBkYbY6.png)|




#### Features comparison
||  munit |Cycle GAN|fast transfer |  Neural Algorithm|dirt| 
|:-----:|:----:|:----:|:----:|:----:|:----:|
|輸入的img類型|content_img + style_img|content_img|content_img + style_img|content_img + style_img|content_img + style_img|
|輸出的img可變性|o|x|o|o|o|
|需要training|o|o|x|x|o|
|轉換效果|筆觸+顏色|筆觸+顏色|顏色|筆觸+顏色|筆觸+顏色|
|轉換速度排名|3rd|2nd|1th|5th|4th|


## Conclusion


- MUNIT 跟 CNN 有類似的照片輸入模式，乍看之下似乎CNN效果比較好?事實上不盡然如此，如果看CNN的realphoto2cezan結果，我們會發現照片內的船幾乎消失不見，在細節保留上是比Munit差很多的，原因推測可能是因為munit在抽取content特徵時，使用了residual block，因此得以保留更多細節。相對而言，因為細節保留太多在很多style轉換後，仍然看起來不像畫作。

- Cycle GAN在細節保留上，似乎取得更好的平衡，因為對其a2b後，尚須進行b2a還原測試，大部分影像轉換後的結果，可細觀出原始照片細節，並且畫風中的筆觸跟色彩皆有融合其中之微妙。
