# import os
# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 
import numpy as np 

def recover_cordi(param):
    # load previously saved quadrilaterals from file and draw them on the image
    count = -1
    cum_vertices = []
    
    with open("slot_vertices.txt", "rb") as f:
        quadrilaterals = f.readlines()

    for i, q in enumerate(quadrilaterals):

        pts = np.array(q.split(), np.int32).reshape(-1, 2)
        cum_vertices.append(pts)
        count += 1

        center_x = int((pts[0][0] + pts[2][0]) / 2)
        center_y = int((pts[0][1] + pts[2][1]) / 2)

        cv2.polylines(param, [pts], True, (0, 255, 0), 2)
        cv2.putText(param, str(i), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(window_name, param)

    return count, cum_vertices

class MouseGesture():
    def __init__(self, count):
        self.is_dragging = False 
        # 마우스 위치 값 임시 저장을 위한 변수 
        self.x0, self.y0, self.w0, self.h0 = -1,-1,-1,-1
        self.points = []
        self.count = count

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(param, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(window_name, param)
            if len(self.points) % 4 == 0 and len(self.points) >= 4:
                self.count += 1
                # draw the quadrilateral
                for i in range(0, len(self.points), 4):
                    pts = np.array(self.points[i:i+4], np.int32)
                    center_x = int((pts[0][0] + pts[2][0]) / 2)
                    center_y = int((pts[0][1] + pts[2][1]) / 2)
                    cv2.polylines(param, [pts], True, (0, 255, 0), 2)
                    cv2.putText(param, str(self.count), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name, param)
                # save the coordinates of all quadrilaterals to a text file
                with open("slot_vertices.txt", "a") as f:
                    for i in range(0, len(self.points), 4):
                        pts = np.array(self.points[i:i+4], np.int32)
                        f.write(" ".join([str(p) for p in pts.flatten()]) + "\n")
                self.points.clear()
        # return 


if __name__=="__main__":

    webcam = cv2.VideoCapture(0)
   
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 노출 보정시키기 위한 값
    for _ in range(5):
        status, frame = webcam.read()
    img = frame
    webcam.release()

    window_name = 'mouse_callback'

    count, cum_vertices = recover_cordi(img)

    mouse_class = MouseGesture(count)


    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, mouse_class.on_mouse, param=img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(cum_vertices)