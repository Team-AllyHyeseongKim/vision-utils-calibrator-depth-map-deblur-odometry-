#!/usr/bin/python
import socket
import cv2
import numpy as np

#socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

#수신에 사용될 내 ip와 내 port번호
TCP_IP = '192.168.1.112'
TCP_PORT = 5001

#remap 파라미터
left_maps0 = np.load("c:\\servertest\\left_maps0.npy")
left_maps1 = np.load("c:\\servertest\\left_maps1.npy")
right_maps0 = np.load("c:\\servertest\\right_maps0.npy")
right_maps1 = np.load("c:\\servertest\\right_maps1.npy")

#TCP소켓 열고 수신 대기
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()

i = 0
while True:
    #String형의 이미지를 수신받아서 이미지로 변환 하고 화면에 출력
    stringData = recvall(conn, 80)
    left_length = recvall(conn, 16)  # 길이 16의 데이터를 먼저 수신하는 것은 여기에 이미지의 길이를 먼저 받아서 이미지를 받을 때 편리하려고 하는 것이다.
    right_length = recvall(conn, 16)
    left_stringData = recvall(conn, int(left_length))
    right_stringData = recvall(conn, int(right_length))

    imu_data = np.frombuffer(stringData, dtype='float')
    data = np.frombuffer(left_stringData, dtype='uint8')
    decimg=cv2.imdecode(data,1)
    left_image = cv2.remap(decimg, left_maps0, left_maps1, cv2.INTER_LANCZOS4)
    data = np.frombuffer(right_stringData, dtype='uint8')
    decimg=cv2.imdecode(data,1)
    right_image = cv2.remap(decimg, right_maps0, right_maps1, cv2.INTER_LANCZOS4)

    cv2.imshow("test1", left_image)
    cv2.imshow("test2", right_image)
    cv2.waitKey(1)

    cv2.imwrite("C:\\scanData\\result\\left\\"+str(i)+".jpg", left_image)
    cv2.imwrite("C:\\scanData\\result\\right\\"+str(i)+".jpg", right_image)
    i += 1



s.close()
cv2.destroyAllWindows()