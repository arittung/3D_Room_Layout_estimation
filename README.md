## 단일 파노라마 입력의 실내 공간 레이아웃 복원 모델 경량화<br>(Lightweight Deep Learning for Room Layout Estimation with a Single Panoramic Image)



<b>Dayoung Kil and Seong-heum Kim</b> | [2022.10 KCI Paper](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002884719), [2022 ICCAS Paper](https://ieeexplore.ieee.org/document/10003901)

<br>

<!--
<b>기존 연구</b> | [sunset1995 연구](https://github.com/sunset1995/HorizonNet)
-->

<br>

<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/185854627-a5ef3eb3-29df-4a34-8896-25c7776c4e58.png" style="width:700px"></p>

<!--
<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/185854576-e3e51c21-f31a-43f2-ba65-baba2f78382b.png" style="width:700px"></p>
-->



<br>

<!--
##  HorizonNet

- Manhattan World 가정을 이용하여 일반적인 RGB 카메라로 촬영된 파노라마 사진 한 장으로 3D 실내 공간을 복원하는 기술

#### 1) 전처리
입력으로 받은 파노라마 사진을 수직 보정하여 소식점과 edge를 찾는다.

#### 2) 특징 추출
ResNet50과 LSTM을 이용하여 훈련된 모델로 사진의 특징을 추출하여 천장-벽 경계, 바닥-벽 경계, 벽-벽 경계가 표시된 1D 레이아웃을 도출한다.

#### 3) Post-processing
Manhattan World 가정으로 바닥과 천장, 벽면을 복구한다. (천장-바닥 거리를 계산한 후 벽면을 복구)

![image](https://user-images.githubusercontent.com/53934639/185855921-a825a842-8bd7-4edb-aeb6-1ed42a78983e.png)
-->



## Proposed Method
we propose a method to lightweight the feature extraction network of HorizonNet by replacing ResNet and LSTM with MnasNet and GRU.

HorizonNet에 사용되는 특징 추출 네트워크를 최적화하여 경량화된 모델로 단일 파노라마 입력에서 실내 공간 복원 방법 제시.

### 1) LSTM -> GRU

### 2) ResNet50 -> MnasNet


<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/185856174-e3c24a46-1594-4c58-beab-de2db472dfa6.png" style="width:700px"></p>


<br>


## Results
### Quantitative Results

The parameters used in the calculation have decreased by more than 1/2.
At the same time, it was confirmed that there was no significant difference from the existing models in the performance measurement of 2D IOU, 3D IOU, MSE, Pixel error, and Corner error. <br>
MnasNet has a great influence on lightweighting, and GRU has an impact on accuracy.

연산 시 사용되는 Parameter가 약 1/2이상 감소함과 동시에, 2D IOU, 3D IOU, MSE, Pixel error, Corner error의 성능 측정에서 기존 모델과 큰 차이를 보이지 않았음을 확인하였다. <br>
MnasNet은 경량화에 큰 영향을 주고, GRU는 정확도에 영향을 준다.



Test casese|ResNet50-LSTM|MnasNet-LSTM|*Our MnasNet-GRU
--|--|--|--
#Parameter (Total) |81,570,348|40,397,700|37,641,092
#FLOPs|71.83|59.19|58.48
2D IOU (%)|87.07|84.16|85.07
3D IOU (%)|84.53|80.80|81.89
RMSE|0.18|0.23|0.21
Pixel error (%)|2.04|2.57|2.70
Corner error (%)|0.65|0.80|0.84


<br>

### Qualitative Results
Similar performance to existing models.

<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/185856772-e56f87ae-99c5-47ba-83c6-f0c3ecbacd5d.png" style="width:900px"></p>





<br><br>

