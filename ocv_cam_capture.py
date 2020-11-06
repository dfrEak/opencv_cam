
import cv2
import ocv
import os

img_PATH='img'

def cam_capture(mirror=False):
    cam = cv2.VideoCapture(0)
    #cam.set(3,720)
    #cam.set(4,1280)
    s, im = cam.read() # captures image
    cv2.imshow("Test Picture", im) # displays captured image
    cv2.imwrite(os.path.join(img_PATH,"test.jpg"),im) # writes image test.jpg to disk
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    cam_capture(mirror=True)


if __name__ == '__main__':
    main()