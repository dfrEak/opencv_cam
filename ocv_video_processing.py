
import cv2
import ocv
import os

img_PATH='img'
vid_PATH='vid'


def vid_open(file_name):
    cap = cv2.VideoCapture(file_name)
    first=True
    change_first=True
    first_frame= None
    while(cap.isOpened()):
        ret,frame = cap.read()
        
        if ret: # check ! (some webcam's need a "warmup")
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('frame',gray)
            
            #save first frame
            if(first==True):
                
                first=False
                cv2.imshow("First Frame", frame) # displays captured image
                #no need to save it 
                #cv2.imwrite(os.path.join(img_PATH,"test.jpg"),frame) # writes image test.jpg to disk
                first_frame=frame
            
            #threshold_img=ocv.laserDetectionVideo('test.jpg',frame)
            threshold_img=ocv.laserDetectionVideo(first_frame,frame)
            threshold,edited_img=ocv.drawObject(threshold_img,img_source=frame)

            if(change_first==True):
                #if false then only use 1 at begining
                #if true then use the previous frame
                change_first=False
                cv2.imshow("First Frame", frame) # displays captured image
                # save the first frame
                first_frame=frame
            
            cv2.imshow('my webcam', edited_img)
            cv2.imshow('threshold', threshold)
        else: #if ret is false, then video is in last frame
            print("video finish")
            while(True):
                if cv2.waitKey(1) == ord('x'): #x for resume again
                    cv2.VideoCapture.set(cap, cv2.CAP_PROP_POS_FRAMES, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) == 32: #space for retake 1st frame
            print("retake first image")
            change_first=True
        elif cv2.waitKey(1) == ord('s'): #s for save
            cv2.imwrite('screenshot.png',edited_img)
            print('ss saved')
        elif cv2.waitKey(1) == ord('p'): #p for playback
            print("playback")
            #set pointer to frame index i 
            #cv2.cvSetCaptureProperty(cap, cv2.CV_CAP_POS_FRAMES, 0)
            cv2.VideoCapture.set(cap, cv2.CAP_PROP_POS_FRAMES, 0)
        elif cv2.waitKey(1) == ord('z'): #z for pause 
            while(True):
                if cv2.waitKey(1) == ord('x'): #x for resume again
                    break
                elif cv2.waitKey(1) == ord('s'): #s for save
                    cv2.imwrite('screenshot.png',edited_img)
                    print('ss saved')

    
    cap.release()
    cv2.destroyAllWindows()


def main():
    #vid_open(os.path.join(vid_PATH,"test3.mp4"))
    #vid_open(os.path.join(vid_PATH,"output_cam1_focused.avi"))
    #vid_open(os.path.join(vid_PATH,"output_cam1_v2.avi"))
    #vid_open(os.path.join(vid_PATH,"output_camkitty_focused.avi"))
    #vid_open(os.path.join(vid_PATH,"output_camkitty_v2.avi"))
    
    #cars
    #vid_open(os.path.join(vid_PATH,"video2.avi"))
    
    #dataset
    #https://motchallenge.net/vis/MOT17-13
    #vid_open(os.path.join(vid_PATH,"AVG-TownCentre.mp4"))
    #vid_open(os.path.join(vid_PATH,"PETS09-S2L1.mp4"))
    vid_open("H:\\video\\seq_2.mp4")
    
    

if __name__ == '__main__':
    main()