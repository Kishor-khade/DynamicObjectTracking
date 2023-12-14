import cv2  # image
import time  # delay
import imutils  # resize

cam = cv2.VideoCapture('Media/video.mp4')  # cam id
time.sleep(1)

firstFrame = None
area = 500

while True:
    # read frame from video
    _, img = cam.read()  
    text = "Normal"
    assert img is not None, "file reading was done"

    # resize the image
    img = imutils.resize(img, width=500)  

    # color 2 Gray scale image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    # smoothen the image
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)  

    # capturing 1st frame on 1st iteration
    if firstFrame is None:
        firstFrame = gaussianImg  
        continue

    # absolute diff b/w 1st nd current frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)

    # Converting the pixel value 
    # if it is smaller than the threshold, it is set to 0, 
    # otherwise it is set to a maximum value. 
    threshImg = cv2.threshold(imgDiff, 70, 255, cv2.THRESH_BINARY)[1] 

    threshImg = cv2.dilate(threshImg, None, iterations=2)

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"
    print(text)
    cv2.putText(img, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("CCTV", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
