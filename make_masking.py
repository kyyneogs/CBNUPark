# importing the module
import cv2
   
# function to display the coordinates of
# of the points clicked on the image 

def click_event(event, x, y, flags, params):

    # checking for right mouse clicks     
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame,(x,y),2,(0,255,0),-1)
        cv2.imshow('image', frame)

        with open('mask_cordi.txt', 'a') as f:
             f.write(f'{x} {y} ')

# driver function
if __name__=="__main__":
    with open('mask_cordi.txt', 'w') as f:
        print('')
    # reading the image
    cap = cv2.VideoCapture(0) # 노트북 캠을 읽는다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 노출 보정시키기 위한 값.
    for _ in range(5):
        _, frame = cap.read()
    cap.release()
    # _, frame = capture.read()
    cv2.imshow("image", frame)    

    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window

    
    cv2.destroyAllWindows()