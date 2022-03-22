# 파노라마 영상을 활용하는 3D 공간 복원 최적화 연구 (3D Room Layout Estimation Using a Single Panoramic Images)

> -[sunset1995 연구](https://github.com/sunset1995) 참고

<br>

### 목차
#### 1.&nbsp; &nbsp;[연구 목적, 기대효과 및 필요성](#1-연구-목적-기대효과-및-필요성)
#### 2.&nbsp; &nbsp;[관련 연구](#2-관련-연구-1)
#### 3.&nbsp; &nbsp;[HorizonNet](#3-horizonnet-1)
#### 4.&nbsp; &nbsp;[네트워크 최적화 방법 제시](#4-네트워크-최적화-방법-제시-1)
#### 5.&nbsp; &nbsp;[최적화 결과](#5-최적화-결과-1)

<br>

--- 
<br>

## 1. 연구 목적, 기대효과 및 필요성
### 📑 연구 목적
- Manhattan World 가정을 이용하여 일반적인 RGB 카메라로 촬영된 파노라마 사진 한 장으로 3D Room을 복원하는 기존 네트워크를 최적화 한다.
  - Manhattan World 가정 : 영상에 나타나는 평면들은 3차원상에서 서로 직교하는 평면들로만 이루어져 있다.

![image](https://user-images.githubusercontent.com/53934639/159414500-1265155d-a313-4adb-b329-4104a93e2aac.png)

### ⚡ 기대효과 및 필요성
- 최근, 온라인 집들이처럼 일반인들이 온라인으로 자신의 집구조와 인테리어를 공유하는 사례가 많아지고 있다, 이러한 공유는 단순 사진이나 동영상 촬영을 통해 이루어 지고 있는데 이러한 경우, 촬영한 각도에 따라 왜곡이 심한 부분이 많고, 실제와 공간의 크기나 비율이 달라 보이는 경우가 비재하다.

- 이 연구는 단순 방의 일부분이 아니라, 전체 구조를 볼 수 있기에 온라인 집들이 뿐 아니라 건축가 및 인테리어 디자이너 혹은 이사를 위해 집 구조를 알아보는 사람들에게 도움이 될 수 있다.

<br>

<br>

## 2. 관련 연구
### 2-1. RoomNet
- 방의 일부 이미지에서 레이아웃을 추정하는 작업.
- end-to-end trainable encoder-decoder Network(RNN)를 통해 키포인트 집합을 추출하고, 획득한 keypoint를 특정 순서로 연결하여 방 레이아웃을 그림.

<img src="https://user-images.githubusercontent.com/53934639/159415313-b0866eb6-8b9b-484c-aab2-78481aea9c57.png" style="width:500px">


<br>

### 2-2. LayoutNet
- 소실점을 기준으로 파노라마 이미지를 정렬한 후, 시스템은 심층네트워크를 사용하여 파노라마 이미지의 boundary와 corner map을 예측한 후 복구하는 작업.

<img src="https://user-images.githubusercontent.com/53934639/159415332-ea394790-7d9b-4a43-84f1-eedf67c24742.png" style="width:500px">

<br>

### 2-3. HorizonNet
- 입력으로 받은 파노라마 사진을 정렬한 후, ResNet50과 LSTM으로 훈련된 모델로 사진의 특징을 추출하여 천장-벽 경계, 바닥-벽 경계, 벽-벽 경계가 표시된 1D Layout을 도출한다. 

- 그 후, Manhattan World 가정으로 바닥, 천장, 벽면을 복구한다.

<img src="https://user-images.githubusercontent.com/53934639/159415347-55719953-a67b-4734-b978-aab51cc755ed.png" style="width:500px">

<br>

### 2-4. HoHoNet
- 레이아웃 재구성을 위한 새로운 방법을 설계하는 것이 아니라, 360 이미지에 대한 깊이 추정과 Semantic segmentation을 모델링하는 작업.

<img src="https://user-images.githubusercontent.com/53934639/159415364-0692f720-3572-430a-aa58-a412588da865.png" style="width:300px">

<br>
<br>

## 3. HorizonNet
- Manhattan World 가정을 이용하여 일반적인 RGB 카메라로 촬영된 파노라마 사진 한 장으로 3D Room을 복원한다.

![image](https://user-images.githubusercontent.com/53934639/159415686-e886fe6a-9c5a-4aaa-9381-23117175fade.png)
![image](https://user-images.githubusercontent.com/53934639/159415832-095cde6b-0bbd-4987-a120-ba505af5217c.png)


<br>
<br>

## 4. 네트워크 최적화 방법 제시
- 기존의 LSTM보다 단순한 구조를 가지는 “GRU구조”로 네트워크를 바꾸어 훈련 시킴.

- LSTM의 forget gate와 input gate를 GRU에서는 하나의 ＂update gate＂로 합침.
  - update gate의 계산 한 번으로 LSTM의 forget gate + input gate의 역할을 대신할 수 있다.
    - Forget gate : 과거 정보를 잊기 위한 게이트
    - Input gate : 현재 정보를 기억하기 위한 게이트
    
- LSTM의 cell state와 hidden state를 GRU에서는 하나의 “hidden state”로 합침. 
    - Cell state : Lstm이 굴러가는 일종의 체인 역할을 하며 기억을 오랫동안 유지할 수 있는 구조
    - Hidden state : 계층의 출력으로 다음 타임 스텝으로 정보를 넘김.

<img src="https://user-images.githubusercontent.com/53934639/159416279-4b7a5f6b-affc-49b9-906c-cf5ba00bb06e.png" style="width:500px">


<br>
<br>

## 5. 최적화 결과
### 5-1. 정량적 결과

- GRU의 파라미터 수가 더 적음. 즉, 기존 Network보다 더 작은 사이즈를 가짐.

- GRU로 변경 시, 훈련 시간이 더 적게 걸림.

- 3D IOU, Corner error, Pixel error 등을 비교해 봤을 때 성능 면에서 큰 차이를 보이지 않음.

![image](https://user-images.githubusercontent.com/53934639/159416592-26e5d592-221d-47b8-8683-a5f381b4fb4e.png)

<br>

### 5-2. 정성적 결과
- 성능면에서도 기존의 결과와 큰 차이를 보이지 않음.

![image](https://user-images.githubusercontent.com/53934639/159416730-3c455360-a6c9-4eb1-ad2d-678a0d435f54.png)



<!--[arittung.log - 3D Room Layout Estimation](https://velog.io/@arittung/series/3D-Room-Reconstruction) 참고.-->

<br><br>

