
import cv2
import ocv
import os

img_PATH='img'


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    #cam.set(3,720)
    #cam.set(4,1280)
    
    #start camera
    first=True
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
            
        #save first frame
        if(first==True):
            first=False
            cv2.imshow("First Frame", img) # displays captured image
            cv2.imwrite(os.path.join(img_PATH,"test.jpg"),img) # writes image test.jpg to disk
        
        threshold_img=ocv.laserDetectionVideo('test.jpg',img)
        threshold,edited_img=ocv.drawObject(threshold_img,img_source=img)

        cv2.imshow('my webcam', edited_img)
        cv2.imshow('threshold', threshold)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        elif cv2.waitKey(1) == 32: #space
            first=True
        elif cv2.waitKey(1) == ord('s'): #s for save
            cv2.imwrite('screenshot.png',edited_img)
            print('ss saved')
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()