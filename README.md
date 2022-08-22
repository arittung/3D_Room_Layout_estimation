## 파노라마 영상을 활용하는 3D 공간 복원 최적화 연구 (3D Room Layout Estimation Using a Single Panoramic Images)

*기존 연구: [sunset1995 연구](https://github.com/sunset1995)*




<br>

### 1. 연구 목적

- 파노라마 사진 한 장으로 3D 실내 공간을 복원하는 기술인 HorizonNet에 사용되는 특징 추출 네트워크를 최적화하여, 단일 파노라마 입력에서 실내 공간을 더 가벼운 모델로 복원하는 방법을 제시하였다.

<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/185854576-e3e51c21-f31a-43f2-ba65-baba2f78382b.png" style="width:500px"></p>


<br>
<br>

### 2. 관련 연구

RoomNet|LayoutNet|HorizonNet|HoHoNet
--|--|--|--
https://github.com/GitBoSun/roomnet|https://github.com/sunset1995/pytorch-layoutnet|https://github.com/sunset1995/HorizonNet|https://github.com/sunset1995/HoHoNet
● 방의 일부 이미지에서 레이아웃을 추정하는 작업.<br>● end-to-end trainable encoder-decoder Network(RNN)를 통해 키포인트 집합을 추출하고, 획득한 keypoint를 특정 순서로 연결하여 방 레이아웃을 그림.|● 소실점을 기준으로 파노라마 이미지를 정렬한 후, 시스템은 심층네트워크를 사용하여 파노라마 이미지의 boundary와 corner map을 예측한 후 복구하는 작업.|● 입력으로 받은 파노라마 사진을 정렬한 후, ResNet50과 LSTM으로 훈련된 모델로 사진의 특징을 추출하여 천장-벽 경계, 바닥-벽 경계, 벽-벽 경계가 표시된 1D Layout을 도출한다. <br>● 그 후, Manhattan World 가정으로 바닥, 천장, 벽면을 복구한다.|● 레이아웃 재구성을 위한 새로운 방법을 설계하는 것이 아니라, 360 이미지에 대한 깊이 추정과 Semantic segmentation을 모델링하는 작업.
<img src="https://user-images.githubusercontent.com/53934639/159415313-b0866eb6-8b9b-484c-aab2-78481aea9c57.png" style="width:500px">|<img src="https://user-images.githubusercontent.com/53934639/159415332-ea394790-7d9b-4a43-84f1-eedf67c24742.png" style="width:500px">|<img src="https://user-images.githubusercontent.com/53934639/159415347-55719953-a67b-4734-b978-aab51cc755ed.png" style="width:500px">|<img src="https://user-images.githubusercontent.com/53934639/159415364-0692f720-3572-430a-aa58-a412588da865.png" style="width:300px">




<br>
<br>

### 3. HorizonNet

- Manhattan World 가정을 이용하여 일반적인 RGB 카메라로 촬영된 파노라마 사진 한 장으로 3D 실내 공간을 복원하는 기술

#### (1) 전처리
입력으로 받은 파노라마 사진을 수직 보정하여 소식점과 edge를 찾는다.

#### (2) 특징 추출
ResNet50과 LSTM을 이용하여 훈련된 모델로 사진의 특징을 추출하여 천장-벽 경계, 바닥-벽 경계, 벽-벽 경계가 표시된 1D 레이아웃을 도출한다.

#### (3) Post-processing
Manhattan World 가정으로 바닥과 천장, 벽면을 복구한다. (천장-바닥 거리를 계산한 후 벽면을 복구)

![image](https://user-images.githubusercontent.com/53934639/185855921-a825a842-8bd7-4edb-aeb6-1ed42a78983e.png)





<br>
<br>

### 4. 네트워크 최적화 방법 제시

<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/185854627-a5ef3eb3-29df-4a34-8896-25c7776c4e58.png" style="width:700px"></p>


#### (1) LSTM -> GRU
- 기존의 LSTM보다 단순한 구조를 가지는 “GRU구조”로 네트워크를 바꾸어 훈련 시킴.


#### (2) ResNet50 -> MnasNet


<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/185856174-e3c24a46-1594-4c58-beab-de2db472dfa6.png" style="width:700px"></p>


<br>
<br>

### 5. 최적화 결과
#### 5-1. 정량적 결과

- 연산 시 사용되는 Parameter가 약 1/2이상 감소함과 동시에, 2D IOU, 3D IOU, MSE, Pixel error, Corner error의 성능 측정에서 기존 모델과 큰 차이를 보이지 않았음을 확인하였다. 
- MnasNet은 경량화에 큰 영향을 주고, GRU는 정확도에 영향을 준다.

<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/185856424-30d0270c-c490-4ad6-abbb-e3458a2c47af.png" style="width:600px"></p>

<br>

#### 5-2. 정성적 결과
- 성능면에서도 기존의 결과와 큰 차이를 보이지 않음.

<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/185856772-e56f87ae-99c5-47ba-83c6-f0c3ecbacd5d.png" style="width:900px"></p>





<br><br>

