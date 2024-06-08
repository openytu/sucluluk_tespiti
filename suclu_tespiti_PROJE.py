import cv2
import time
import mediapipe as mp
import math
import face_recognition

resim_1_path = ""
resim_2_path = ""
resim1Image= face_recognition.load_image_file(resim_1_path)
resim2Image = face_recognition.load_image_file(resim_2_path)
resim1ImageEncodings=face_recognition.face_encodings(resim1Image)[0]
resim2ImageEncodings=face_recognition.face_encodings(resim2Image)[0]
encodingList= [resim1ImageEncodings, resim2ImageEncodings]
nameList=["Şüpheli 1", "Şüpheli 2"]

def calculate_FPS(cTime, ptime):
    # FPS hesabı
    fps = 1 / (cTime - ptime)
    ptime = cTime
    return fps, ptime
def resim_cek(name_o, frame):

    face_locations = face_recognition.face_locations(frame)
    faceEncodings= face_recognition.face_encodings(frame,face_locations)
    for face_location, faceEncoding in zip(face_locations, faceEncodings):
        topy, rightx, bottomy, leftx = face_location
        detectedFaces = frame[ topy:bottomy,leftx:rightx]   

        matchedFaces = face_recognition.compare_faces(encodingList,faceEncoding)
        name = "unknown"
        if True in matchedFaces:
            matchedIndex= matchedFaces.index(True)
            name=nameList[matchedIndex]    
        
        cv2.rectangle(frame, (leftx, topy), (rightx, bottomy), (0, 255, 255), 3)
        img_path = "your_results_path" + "/" + name + "_" + name_o
        cv2.imwrite(img_path, detectedFaces)
        print(f"{name} adlı şüphelinin görüntüsü {img_path} olarak kaydedildi.")    

def mesafe(x1, y1, x2, y2):
    d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return d

font = cv2.FONT_HERSHEY_PLAIN
color1 = (255, 0, 0)
cap = cv2.VideoCapture(0)


mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils
ptime = 0


coor_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("kapandi")
        break

    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_RGB)
    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHand.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    if len(lmList) > 1:
            x,y = lmList[12][1:]
            coor_list.append((x,y))    
            print("coor list: ",coor_list)
            if len(coor_list) > 1:
                print("len coor list: ",len(coor_list))
                uzaklik = mesafe(coor_list[0][0], coor_list[0][1], coor_list[1][0], coor_list[1][1])
                print("uzaklik: ",uzaklik)
                coor_list.pop(0)
                print("after coor list: ",coor_list)
                if uzaklik > 250:
                    current_time = time.localtime()
                    formatted_time = time.strftime("%H-%M-%S", current_time)
                    resim_cek(f"resim_{formatted_time}_{int(uzaklik)}.jpg", frame)
            else:
                continue
    
    cTime=time.time()
    fps, ptime = calculate_FPS(cTime, ptime)
    cv2.putText(frame, "FPS: " + str(int(fps)), (10, 75), font, 3, color1, 5)

    cv2.imshow("pencere", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
