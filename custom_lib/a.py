import cv2

cap = cv2.VideoCapture(1)
cap.set(3, 480)
cap.set(4, 640)

cup = cv2.VideoCapture(2)
cup.set(3, 480)
cup.set(4, 640)
i = 0
while True:
    ret, frame = cap.read()
    ret2, frame2 = cup.read()
    cv2.imshow('test1', frame)
    cv2.imshow('tast2', frame2)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == 32:
        cv2.imwrite("C:\\calib\\left\\"+str(i)+".jpg", frame)
        cv2.imwrite("C:\\calib\\right\\"+str(i)+".jpg", frame2)
        i += 1

cap.release()
cup.release()

cv2.destroyAllWindows()

