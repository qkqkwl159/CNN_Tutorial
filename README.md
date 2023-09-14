# CNN_Tutorial

studing Machine Learning

## 1. 라이브러리 및 데이터셋 불러오기
- numpy
배열 및 수학 연산을 위해 사용 
- tensorflow
딥러닝과 관련된 대부분의 기능을 제공

- CIFAR-10
10개의 다른 클래스(자동차, 비행기, 개, 고양이 등) 속하는 6만개의 32x32 컬러 이미지로 구성된
데이터 셋, training 5천개와 testing 1만개로 나뉜다.

## 2. CNN 모델 구성하기
- 컨볼루션 레이어(Conv2D)
이미지의 특징을추출하(Feature Extraction)사용.
이레이어는 필터를 사용하여 이미지의 여러부분에 걸쳐 슬라이딩하며 특징맵을 형성.
- 풀링 레이어(MaxPooling2D)
특징 맵의 크기를 줄이며(Down Samplig), 계산량을 감소시키고 중요한 특징을 유지
- 완전 연결 레이어(Densce)
네트워크의 마지막 부분에서 사용되며, 분류(Classfication)을 위한 레이어

## 3. Model Compile
- 옵티마이저
학습률을 자동으로 조정하는 효과적인 최적화 알고리즘중 하나. 
현재 Tutorial에서는 "Adam" 옵티마이저를 사용
- 손실 함수(Loss function)
분류 문제에는 일반적으로 크로스 엔트로피 손실 함수가 사용됩니다. 
SparseCategoricalCrossentropy를 사용하므로 원-핫 인코딩 없이 정수 레이블을 그대로 사용할 수 있다.

## 4. Model 학습
모델을 훈련 데이터에 적합시키며, 각 에포크 후에 검증 데이터를 사용하여 모델의 성능을 평가한다.

## 5. 성능 평가
- evaluate 메서드를 사용
테스트 데이터셋에서 모델의 성능을 평가. 
반환된 값은 손실과 정확도 이다.

