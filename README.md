# 영상처리프로그래밍 기말 프로젝트
---------------------------

## 개요
- opencv와 파이썬을 이용한 얼굴 보정 프로그램
- cascadeclassifier을 이용해 영상 속 얼굴, 눈, 입 분석 후 각 부위별 적절한 보정
- jpg확장자의 이미지 파일, 웹캠 실시간 영상 모두 동일하게 적용 가능


## 주요 기능
- cascadeclassifier을 이용한 얼굴 분석
- 히스토그램 평활화
- 컨볼루션 연산
  > 얼굴 영역에는 매끈한 피부로 보이게끔 평균 필터 적용
  > 눈 영역에는 또렷한 눈으로 보이도록 샤프닝 필터 적용
- 컬러 공간을 활용한 색상 보정
  > cv2.split()을 사용하여 컬러 영상의 각 채널 분리 후 붉은 색의 R채널의 값을 증가시켜 생기있는 입술로 보이도록 함


### 웹캠에 적용하였을 때
<img width="500" alt="image" src="https://github.com/azzbc7819/imageprocessing_termProject/assets/80818761/b9f749c7-861d-4f79-9da8-52ae4ff5d570">

### jpg 확장자 이미지에 적용하였을 때
![image](https://github.com/azzbc7819/imageprocessing_termProject/assets/80818761/eea54b92-dd1b-4888-a603-a5d44226bbe1)


