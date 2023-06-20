import cv2, numpy as np
from Common.filters import filter

def preprocessing(imgname):
    image = cv2.imread('images/%s.jpg' %imgname, cv2.IMREAD_COLOR) #이미지 가져오기
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #흑백 영상 가져오기
    gray = cv2.equalizeHist(gray) #히스토그램 평활화
    return image, gray

def preprocessing2(frame):
    if frame is None: return None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #흑백 영상 가져오기
    gray = cv2.equalizeHist(gray) #히스토그램 평활화
    return frame, gray

def define_roi(pt, size):
    return np.ravel((pt, size)).astype('int')

def detect_object(center, face): #입술 영역 검출
    w, h = np.array(face[2:4]) #얼굴 영역의 너비와 높이 가져오기
    center = np.array(center) #얼굴 중심
    lip_center = center + (0, int(h * 0.3)) #입술 중심
    lip_w = w * 0.18 #입술 중심에서부터 입술 끝까지의 가로 길이
    lip_h = h * 0.1 #입술 중심에서부터 임술 끝까지의 세로 길이
    gap2 = (lip_w, lip_h)

    lip1 = lip_center - gap2
    lip2 = lip_center + gap2
    lip = define_roi(lip1, lip2-lip1)

    return lip

def webcam_correction(capture, face_cascade, eye_cascade):
    #웹캠을 이용한 경우 보정
    while True:
        ret, frame = capture.read()  # 웹캠 영상의 신호와 frame 행렬 가져오기
        if not ret: break

        image, gray = preprocessing2(frame)
        correction_image = image.copy() #보정될 이미지 카피

        faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100)) #얼굴 영역 검출

        # print(faces)
        if faces == ():
            print("none")

            cv2.imshow("original image", image)
            cv2.imshow("correct image", correction_image)
            if cv2.waitKey(30) >= 0: break
            continue
        elif faces.any():
            x, y, w, h = faces[0]
            face_image = image[y:y + h, x:x + w] #입력 영상에서의 검출된 얼굴 영역
            # face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            blurface = cv2.filter2D(face_image, -1, mask1) #얼굴 영역에 컨볼루션 연산
            correction_image[y:y + h, x:x + w] = blurface #복사된 이미지의 얼굴 영역에 컨볼루션 연산한 블러 이미지로 바꿔주기
            # cv2.imshow("blurface", blurface)
            # cv2.imshow("not blur face", face_image)
            eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20)) #눈 영역 검출
            if len(eyes) == 2:
                face_center = (x + w // 2, y + h // 2) #얼굴 중심
                lip = detect_object(face_center, faces[0]) #입술 영역 검출
                lip_image = image[lip[1]:lip[1] + lip[3], lip[0]:lip[0] + lip[2]] #입력 영상에서의 검출된 입술 영역
                blip, glip, rlip = cv2.split(lip_image) #채널 분리
                rlip = cv2.add(rlip, 10) #r채널에서의 화소값 증가
                correct_lip = cv2.merge((blip, glip, rlip)) #증가시킨 영상 다시 합성
                correction_image[lip[1]:lip[1] + lip[3], lip[0]:lip[0] + lip[2]] = correct_lip #복사된 이미지에 색 보정된 이미지로 입술 영역 바꿔주기
                # cv2.imshow("correct lip", correct_lip)
                # cv2.rectangle(image, lip, (255, 0, 0), 2)

                for ex, ey, ew, eh in eyes:
                    center = (x + ex + ew // 2, y + ey + eh // 2)
                    eye_pos = image[y + ey:y + ey + eh, x + ex:x + ex + eh]  # 눈 위치
                    sharp_eye = cv2.filter2D(eye_pos, -1, mask2) #눈 위치에 샤프닝 필터 컨볼루션 연산
                    correction_image[y + ey:y + ey + eh, x + ex:x + ex + eh] = sharp_eye #복사된 이미지의 눈 영역 보정된 이미지로 변경
                    # cv2.imshow("eye", eye_pos)
                    # cv2.imshow("sharp eye", sharp_eye)
                    # cv2.circle(image,center, 10, (0,255,0), 2) #이 서클 자리에 딱 샤프닝 필터가 씌워지면 좋은데

            else:
                print('눈 미검출')
            # cv2.rectangle(image, faces[0], (255, 0,0), 2)

        else:
            print("얼굴 미검출")

        cv2.imshow("original image", image)
        cv2.imshow("correct image", correction_image)
        if cv2.waitKey(30) >= 0: break

    capture.release()

def jpg_correction(filename, face_cascade, eye_cascade):
    #jpg 이미지를 이용한 경우의 보정
    image, gray = preprocessing(filename)
    if image is None: raise Exception("영상파일 읽기 에러")

    correction_image = image.copy() #보정될 이미지 카피

    faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100)) #얼굴 영역 검출

    if faces.any():
        x, y, w,h = faces[0]
        face_image = image[y:y+h, x:x+w]
        blurface = cv2.filter2D(face_image, -1, mask1) #컨볼루션 연산을 통해 얼굴 영역 블러 효과
        correction_image[y:y + h, x:x + w] = blurface

        eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25,20)) #눈 영역 검출


        if len(eyes) == 2:
            face_center = (x + w // 2, y + h // 2) #얼굴 중심
            lip = detect_object(face_center, faces[0]) #입술 영역 검출
            lip_image = image[lip[1]:lip[1] + lip[3], lip[0]:lip[0] + lip[2]]
            #lip_image = cv2.cvtColor(lip_image, cv2.COLOR_BGR2HSV)
            blip, glip, rlip = cv2.split(lip_image) #입술 색상 보정을 위한 채널 분리
            #hlip, slip, vlip = cv2.split(lip_image)
            #hlip = cv2.subtract(hlip, 10)
            #correct_lip = cv2.cvtColor(cv2.merge((hlip, slip, vlip)), cv2.COLOR_HSV2BGR)
            rlip = cv2.add(rlip, 10) #R채널 화소값 증가
            correct_lip = cv2.merge((blip, glip, rlip)) #단일 채널로 합병
            correction_image[lip[1]:lip[1] + lip[3], lip[0]:lip[0] + lip[2]] = correct_lip

            for ex, ey, ew, eh in eyes:
                center = (x + ex + ew // 2, y + ey + eh // 2)
                eye_pos = image[y + ey:y + ey + eh, x + ex:x + ex + eh]  # 눈 위치
                sharp_eye = cv2.filter2D(eye_pos, -1, mask2) #샤프닝 필터로 컨볼루션 연산
                correction_image[y + ey:y + ey + eh, x + ex:x + ex + eh] = sharp_eye
        else:
            print("눈 미검출")

        cv2.imshow("origin image", image)
        cv2.imshow("correction image", correction_image)
        cv2.waitKey(0)
    else:
        print("얼굴 미검출")





data1 = [1/9, 1/9, 1/9,
         1/9, 1/9, 1/9,
         1/9, 1/9, 1/9]

data2 = [ 0, -1, 0,
          -1, 5, -1,
          0, -1, 0]

mask1 = np.array(data1, np.float32).reshape(3,3) #얼굴 블러를 위한 필터
mask2 = np.array(data2, np.float32).reshape(3,3) #눈 샤프닝을 위한 필터

capture = cv2.VideoCapture(0) #카메라 연결, 웹캠 영상 가져오기
if capture.isOpened() == False:
    raise Exception("카메라 연결 안 됨")

face_cascade = cv2.CascadeClassifier('Common/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('Common/haarcascade_eye.xml')


jpg_correction("shuhwa", face_cascade, eye_cascade)
webcam_correction(capture, face_cascade, eye_cascade)
#jpg_correction("shuhwa", face_cascade, eye_cascade)