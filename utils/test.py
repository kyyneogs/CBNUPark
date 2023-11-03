from parking import readMaskVertices, masking
import cv2

array = readMaskVertices('mask_cordi.txt')

print(array)
cap = cv2.VideoCapture(0) # 노트북 캠을 읽는다.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 노출 보정시키기 위한 값.
for _ in range(5):
    _, frame = cap.read()
cap.release()

frame = masking(frame, array)
# _, frame = capture.read()
cv2.imshow("image", frame)    

# wait for a key to be pressed to exit
cv2.waitKey(0)

# close the window

cv2.destroyAllWindows()