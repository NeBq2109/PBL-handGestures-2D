from pypylon import pylon
import cv2
import numpy as np
import datetime
import functions as hf
from matplotlib import pyplot as plt

import pickle

#elo
import socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('127.0.0.1', 8089)
client.connect(server_address)


def sendMsg(msg):
    message = pickle.dumps(msg)
    sendLength = bytes(str(len(message)), 'utf-8')
    sendLength += b' ' * (header_size-len(sendLength))
    message = sendLength + message
    client.send(message)


# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=20, detectShadows=False)
p=1
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        raw_image = image.GetArray()
        image_roi = raw_image[200:800,200:1000].copy()   #PBL camera
        #image_roi = raw_image[600:2100,600:2700].copy()   #AVS camera
        #image_hist = raw_image[400:1000,200:800].copy()
        
         # Subtractor
        skin_sub,sub_image = hf.bgSubtractor(image_roi.copy(), bgSubtractor)
   
        raw_rec = raw_image.copy()
        raw_rec=cv2.rectangle(raw_rec, (200,200), (1000,800), (0, 255, 0), 10) 
        
        HSV_image = hf.skin_extract(image_roi)

        filled_image = hf.Fill_Holes(HSV_image)
        sub_HSV = sub_image.copy()
        sub_HSV = cv2.bitwise_and(sub_HSV,filled_image)


        contours, length = hf.FB_contours(sub_HSV)
        drawing = sub_HSV.copy()
        if length>0:
            maxcontour = hf.Find_MAX_CONTOUR(contours, length)
            drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)
            drawing = cv2.drawContours(drawing, [maxcontour], 0, (0,255,0), 10)
            drawing, _, deffect_count = hf.contour_points(maxcontour, drawing)

            count_string = str(deffect_count)+'\r\n'
            client.send(count_string.encode())

        cv2.namedWindow('raw_image', cv2.WINDOW_NORMAL)
        cv2.imshow('raw_image', raw_rec)
        
        cv2.namedWindow('image_roi', cv2.WINDOW_NORMAL)
        cv2.imshow('image_roi', image_roi)     

        '''
        cv2.namedWindow('skin_sub_image', cv2.WINDOW_NORMAL)
        cv2.imshow('skin_sub_image', skin_sub)
        '''
        cv2.namedWindow('HSV_image', cv2.WINDOW_NORMAL)
        cv2.imshow('HSV_image', HSV_image)
        
        cv2.namedWindow('bg_subtractor', cv2.WINDOW_NORMAL)
        cv2.imshow('bg_subtractor', sub_image)
        
        '''
        cv2.namedWindow('filled_image', cv2.WINDOW_NORMAL)
        cv2.imshow('filled_image', filled_image)
        '''
        cv2.namedWindow('sub_HSV', cv2.WINDOW_NORMAL)
        cv2.imshow('sub_HSV', sub_HSV)
    
        cv2.namedWindow('drawing', cv2.WINDOW_NORMAL)
        cv2.imshow('drawing', drawing)
        k=cv2.waitKey(1)
        '''
        if k == ord('h'):
            p=1          
            handHist = cv2.calcHist([image_hist], [0, 1], None, [180, 256], [0, 180, 0, 256])
            handHist_norm = cv2.normalize(handHist, handHist, 0, 255, cv2.NORM_MINMAX
        if p==1:

            hist_image = hf.hist_extract(image_roi, handHist_norm)
    
            cv2.namedWindow('hist_image', cv2.WINDOW_NORMAL)
            cv2.imshow('hist_image', hist_image)
         '''
        if k == ord('s'):
            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            raw_name = 'prezentacja1/{0}_raw.bmp'.format(date)
            skin_sub_name = 'prezentacja1/{0}_skin_sub_color.bmp'.format(date)
            YCrCb_name = 'prezentacja1/{0}_HSV.bmp'.format(date)
            subtractor_name = 'prezentacja1/{0}_sb.bmp'.format(date)
            sub_ycrb_name = 'prezentacja1/{0}_sb_HSV.bmp'.format(date)
            filled_name = 'prezentacja1/{0}_filled.bmp'.format(date)
            drawing_name = 'prezentacja1/{0}_drawing.bmp'.format(date)
            cv2.imwrite(raw_name,cv2.resize(image_roi,None ,fx=0.2,fy=0.2))
            cv2.imwrite(skin_sub_name, cv2.resize(skin_sub,None ,fx=0.2,fy=0.2))
            cv2.imwrite(YCrCb_name, cv2.resize(HSV_image,None ,fx=0.2,fy=0.2))
            cv2.imwrite(subtractor_name, cv2.resize(sub_image,None ,fx=0.2,fy=0.2))
            cv2.imwrite(sub_ycrb_name, cv2.resize(sub_HSV,None ,fx=0.2,fy=0.2))
            cv2.imwrite(filled_name,  cv2.resize(filled_image,None ,fx=0.2,fy=0.2))
            cv2.imwrite(drawing_name, cv2.resize(drawing,None ,fx=0.2,fy=0.2))           
        if k == ord('q'): break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()
