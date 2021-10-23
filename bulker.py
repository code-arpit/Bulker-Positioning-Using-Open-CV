import cv2 as cv

video = cv.VideoCapture(0)
bulker_cascade = cv.CascadeClassifier('/home/arpitjain/Desktop/Code/tabxolabs/Bulker_Positioning/bulker_cascade.xml')


while True:
    ret, frame = video.read()
    # frame = cv.imread('/home/arpitjain/Desktop/Code/tabxolabs/Bulker_Positioning/positives/jrg67jh7.jpg')
    frame = cv.resize(frame, (700,500))
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detections = bulker_cascade.detectMultiScale(frame_gray,
                                            scaleFactor=3,
                                            minNeighbors=10,
                                            minSize=(90,90),
                                            flags=cv.CASCADE_SCALE_IMAGE)
    
    # cv.rectangle(frame, (r_x,r_y), (r_w,r_h), (0,0,0), thickness=3)
    cv.circle(frame, (350,250), 150, (0,0,0), 4)


    if(len(detections) > 0):
        (t_x,t_y,t_w,t_h) = detections[0]
        if (t_x<=180) :
            cv.rectangle(frame, (t_x,t_y), (t_x+t_w,t_y+t_h), (0,0,255), thickness=3)
            cv.putText(frame, 'Move Forward', (250,80), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        elif ((t_x+t_w)>=520):
            cv.rectangle(frame, (t_x,t_y), (t_x+t_w,t_y+t_h), (0,0,255), thickness=3)
            cv.putText(frame, 'Move Backward', (250,80), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        else:
            cv.rectangle(frame, (t_x,t_y), (t_x+t_w,t_y+t_h), (0,255,0), thickness=3)
            cv.putText(frame, 'STOP!', (250,80), cv.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
        print(f'Number of faces detected = {len(detections)}')
    elif (len(detections) == 0):
        print('Number of faces detected = 0')   
    

    cv.imshow('frame', frame)
    if cv.waitKey(10) & 0xFF==ord('q'):
        break


video.release()
cv.destroyAllWindows()