import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
#from PIL import Image as pillow

img_PATH='img'


"""
# Load an color image in grayscale if 0
img = cv2.imread(os.path.join(img_PATH,'2_20.jpg'),1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

"""

def laserDetection(image0,image1):
    img1=cv2.imread(os.path.join(img_PATH,image1),1)
    return(laserDetectionVideo(image0,img1))

def laserDetectionVideo(image0,img1):

    #img0=cv2.imread(os.path.join(img_PATH,image0),1)
    img0=image0
    #cv2.imshow('img0',img0)
    #cv2.imshow('img1',img1)
    #print(img0.shape[:2])
    #print(img1.shape[:2])

    #find the difference
    diff_img1=cv2.subtract(img1,img0)
    diff_img2=cv2.subtract(img0,img1)
    diff_img=cv2.add(diff_img1,diff_img2)

    cv2.imshow('Diff',diff_img)
    ######################################################################
    ################Method 1: Convert Method
    
    #1. rgb to gray
    bnw_img=diff_img.copy()
    bnw_img=cv2.cvtColor(bnw_img, cv2.COLOR_BGR2GRAY)
    
    #2. gray to BnW
    try:
        thresh=bnw_img[bnw_img.nonzero()].mean()
    except:
        thresh=0
    thresh=thresh+10/100*255
    (thresh, bnw_img) = cv2.threshold(bnw_img, thresh, 255, cv2.THRESH_BINARY)
    
    #cv2.imshow('Method 1',bnw_img)
    
    #test
    
    # closing
    kernel = np.ones((15,15),np.uint8)
    bnw_img = cv2.morphologyEx(bnw_img, cv2.MORPH_CLOSE, kernel)
    # opening
    kernel = np.ones((7,7),np.uint8)
    bnw_img = cv2.morphologyEx(bnw_img, cv2.MORPH_OPEN, kernel)
    # closing
    kernel = np.ones((7,7),np.uint8)
    bnw_img = cv2.morphologyEx(bnw_img, cv2.MORPH_CLOSE, kernel)

    bnw_img=cv2.cvtColor(cv2.bitwise_not(bnw_img), cv2.COLOR_GRAY2BGR)
    cv2.imshow('Test',cv2.add(bnw_img,img1))
    
    
    ######################################################################
    ################Method 2: Red channel threshold
    
    #1. extract red channel by making the other channel 0
    red_img=diff_img.copy()
    red_img[:,:,0]=0 # blue
    red_img[:,:,1]=0 # green
    
    #2. rgb to gray
    red_img=cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('image red',red_img)

    #3. gray to BnW
    try:
        thresh=red_img[red_img.nonzero()].mean()
    except:
        thresh=0
    thresh=thresh+10/100*255
    (thresh, red_img) = cv2.threshold(red_img, thresh, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Method 2',red_img)

    '''
    # gray to BnW

    #(thresh, diff_img) = cv2.threshold(diff_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #fixed amount, can be changed
    #thresh = 50
    #diff_img = cv2.threshold(diff_img, thresh, 255, cv2.THRESH_BINARY)[1]

    # approach 1 
    # source https://stackoverflow.com/questions/7624765/converting-an-opencv-image-to-black-and-white
    #print(diff_img.mean())
    #thresh=diff_img.mean()+10/100*255

    thresh=diff_img[diff_img.nonzero()].mean()
    thresh=thresh+10/100*255
    #thresh=thresh+5/100*(255-thresh)
    #print(thresh)
    threshold_img = cv2.threshold(diff_img, thresh, 255, cv2.THRESH_BINARY)[1]

    #cv2.imshow('image threshold',threshold_img)
   
    
    # erode
    #threshold_img = cv2.erode(threshold_img,kernel,iterations = 1)
    # opening
    kernel = np.ones((2,2),np.uint8)
    threshold_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)
    # closing
    kernel = np.ones((2,2),np.uint8)
    threshold_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow('image result red',threshold_img)


    # make into black and white
    threshold_img[:,:,0]=threshold_img[:,:,2] # blue
    threshold_img[:,:,1]=threshold_img[:,:,2] # green

    #cv2.imshow('image result BW',threshold_img)
    threshold_img=cv2.cvtColor(threshold_img, cv2.COLOR_BGR2GRAY);
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 
    '''

    ######################################################################
    #########Simple BnW Method
    #threshold_img=np.copy(diff_img)
    #threshold_img=cv2.cvtColor(threshold_img, cv2.COLOR_BGR2GRAY)
    #ret,threshold_img=cv2.threshold(threshold_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #ret,threshold_img=cv2.threshold(threshold_img, 10, 255, cv2.THRESH_BINARY)
    
    ######################################################################
    #########Method 3:Filtering Method
    #([17, 15, 100], [50, 56, 200])
    #all
    #lowerBound=np.array([0, 0, 0])
    #upperBound=np.array([100, 255, 255])
    #default
    #lowerBound=np.array([0, 0, 50])
    #upperBound=np.array([60, 60, 200])
    #experiment
    lowerBound=np.array([0, 0, 40])
    upperBound=np.array([60, 60, 255])
    threshold_img=cv2.inRange(diff_img,lowerBound,upperBound)
    
    # erode
    #threshold_img = cv2.erode(threshold_img,kernel,iterations = 1)
    # opening
    kernel = np.ones((3,3),np.uint8)
    #threshold_img = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel)
    # closing
    kernel = np.ones((3,3),np.uint8)
    #threshold_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)
    
    #cv2.imshow('Method 3',bnw_img)
    
    return(threshold_img)

    
def drawObject(threshold_img, img_source=None):
    #https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours

    # find contours and get the external one
    #image, contours, hier = cv2.findContours(threshold_img, cv2.RETR_TREE,
    #                cv2.CHAIN_APPROX_SIMPLE)
    image, contours, hier = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    list=[]
    
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        #cv2.rectangle(threshold_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.rectangle(threshold_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #cv2.rectangle(img1,(int(x),int(y)),(int(x)+int(w),int(y)+int(h)),(0,255,0),3)
     
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        #print(box)
        
        """
        # euclidean distance (not yet)
        p1,p2,p3,p4=box
        #print(p1)
        # p1 and p2
        length=np.linalg.norm(p1-p2)
        # p2 and p3
        width=np.linalg.norm(p2-p3)
        if(width>length):
            temp=length
            length=width
            width=temp
        #print("length "+str(length))
        #print("width "+str(width))
        # add text (size) to image
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        position               = (x,y)
        fontScale              = 0.5
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(threshold_img,str(length)+'x'+str(width), 
            position, 
            font, 
            fontScale,
            fontColor,
            lineType)
        """
        
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        #cv2.drawContours(img, [box], 0, (0, 0, 255))
        #cv2.drawContours(threshold_img, [box], 0, (255, 0, 0))
        #cv2.drawContours(img1, [box], 0, (255, 0, 0))
     
        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        
        #threshold value
        threshold_area_min=0
        threshold_area_max=10000
        
        #print(cv2.contourArea(c))
        # and draw the circle in blue
        #if(threshold_area_min<cv2.contourArea(c) and threshold_area_max>cv2.contourArea(c)):
        #cv2.circle(threshold_img, center, radius, (255, 0, 0), 2)
        #cv2.circle(img_source, center, radius, (255, 0, 0), 2)
        
        #add to list
        #print(str(cv2.contourArea(c)))
        #if(threshold_area_min<cv2.contourArea(c) and threshold_area_max>cv2.contourArea(c)):
        #   list.append(center)
        
        #img1 =cv2.rectangle(img1,(int(x)-int(radius),int(y)-int(radius)),(int(x)+int(radius),int(y)+int(radius)),(0,255,0),3)

        """
    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(output, contours, -1, 255, 3)

        #find the biggest area
        c = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)
        # draw the book contour (in green)
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2) 
     """
     
        #if(img_source.any()!=None):
            #cv2.drawContours(img_source, [box], 0, (255, 0, 0))
            #cv2.putText(img_source,str(length)+'x'+str(width), 
            #position, 
            #font, 
            #fontScale,
            #fontColor,
            #lineType)
        #    cv2.circle(img_source, center, radius, (255, 0, 0), 2)
    
    #list.append((0,0))
    #print(list)
    
    #draw a line between center
    for i in range(0,len(list)-1):
        dist=np.linalg.norm(np.subtract(list[i],list[i+1]))
        #print("----------------")
        #print(list[i])
        #print(list[i+1])
        #print(np.subtract(list[i],list[i+1]))
        near=i+1
        for j in range(i+1,len(list)):
            if(dist>np.linalg.norm(np.subtract(list[i],list[i+1]))):
                dist=np.linalg.norm(np.subtract(list[i],list[i+1]))
                near=j
                print("dist",dist)
        cv2.line(img_source,list[i],list[near],(255,0,0),2)
        cv2.line(threshold_img,list[i],list[near],(255,0,0),2)
        print("near ",near," dist ",dist)
        
        #add text
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        position               = (int((list[i][0]+list[near][0])/2),int((list[i][1]+list[near][1])/2))
        fontScale              = 0.5
        fontColor              = (0,255,0)
        lineType               = 2

        cv2.putText(img_source,
            str(i)+" to "+str(near)+"=>"+str(np.linalg.norm(np.subtract(list[i],list[i+1]))), 
            position, 
            font, 
            fontScale,
            fontColor,
            lineType)
    
    #print(len(contours))
    cv2.drawContours(threshold_img, contours, -1, (255, 255, 0), 1)
    
    #if(img_source.any()!=None):
    #    cv2.drawContours(img_source, contours, -1, (255, 255, 0), 1)
    return(threshold_img,img_source)   
    
    print(list)
 
def main():
    img_source='test_0_0.jpg'
    threshold_img=laserDetection('test_0_0.jpg','test_0_1.jpg')
    img1=cv2.imread(os.path.join(img_PATH,img_source),1)
    edited_img,img1=drawObject(threshold_img, img_source=img1)

    cv2.imshow("contours", edited_img)
    cv2.imshow("contours1", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    main()



