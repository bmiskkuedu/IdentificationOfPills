'''
요약 : Localization 역할
이미지 하나에 알약의 앞면, 뒷면이 같이 나와 있는 경우
    앞면과 뒷면을 각각 잘라내어 저장한다.
'''

import cv2 as cv
from Utils import PositionManager
import numpy as np


def CropImg(grabedImg, orgImg, contour, fileName, ShowFlag):

    # 전체 이미지의 크기
    imgY = grabedImg.shape[0]
    imgX = grabedImg.shape[1]

    # 현재 컨투어의 좌표
    minX, minY, maxX, maxY, cx, cy = PositionManager.GetMinMaxPosInContour(contour, '')

    width = maxX - minX
    height = maxY - minY

    diffValue = abs(width - height)

    # 정사각형 배경 안에 전경 이미지 영역을 최대한 크게 한다.
    offsetValue = 15
    if width > height:
        xOffset = offsetValue
        yOffset = diffValue // 2 + offsetValue
    elif width < height:
        xOffset = diffValue // 2 + offsetValue
        yOffset = offsetValue
    else:
        xOffset = offsetValue
        yOffset = offsetValue

    initX = minX - xOffset
    if initX < 0:
        initX = 0

    maxX = maxX + xOffset

    # 전체 이미지의 크기를 벗어나지 않게 조절 한다.
    if maxX > imgX:
        maxX = imgX

    initY = minY - yOffset

    if initY < 0:
        initY = 0

    maxY = maxY + yOffset
    if maxY > imgY:
        maxY = imgY

    xLength = maxX - initX
    yLength = maxY - initY
    diffXY = abs(xLength - yLength)
    if diffXY is 0:
        if xLength > yLength:
            maxY = maxY + diffXY
        else:
            maxX = maxX + diffXY

    ImgMasking = np.zeros((grabedImg.shape[0], grabedImg.shape[1], 3), np.uint8)
    cv.drawContours(ImgMasking, [contour], 0, (255, 255, 255), 0)

    # 내부를 흰색으로 채워 줌
    mask = np.zeros((ImgMasking.shape[0] + 2, ImgMasking.shape[1] + 2), np.uint8)
    mask[:] = 0
    cv.floodFill(ImgMasking, mask, (cx, cy), (255, 255, 255))

    #print(grabedImg.shape[1], grabedImg.shape[0])
    #print(orgImg.shape[1], orgImg.shape[0])
    # orgImg를 grabedImg와 크기를 동일하게 맞춰 줌
    #orgImg = cv.resize(orgImg, (grabedImg.shape[1], grabedImg.shape[0]))
    orgImg = grabedImg
    if ShowFlag:
        cv.imshow("ImgMasking1", ImgMasking)
        cv.imshow("orgImg", orgImg)
        cv.waitKey(0)

    # orgImg에서 원하는 위치를 마스킹 함.
    dst = cv.copyTo(orgImg, ImgMasking)
    if ShowFlag:
        cv.imshow("dst", dst)
        cv.waitKey(0)

    # 마스킹된 원본 이미지에서 위에서 구한 좌표로 이미지를 잘라낸다.
    finalImg = dst[initY:maxY, initX:maxX]

    if ShowFlag:
        print(initX, maxX, initY, maxY)
        print(imgX, imgY)
        cv.imshow('crop', finalImg)
        cv.waitKey(0)

    SavePath = 'E:\\Study\\Pill\\Firstdis_images\\image\\Class5\\'
    SaveFlag = True
    if cx < (imgX // 2) :
        SavePath = SavePath + 'Front\\' + fileName
        print("front")
    else :
        print("back")
        SavePath = SavePath + 'Back\\' + fileName
        #SaveFlag = False
        ''' Circle 3개씩 있는 것의 후면 추출 용도(후면이 이미지 왼쪽 중앙, 왼쪽 끝은 옆면이미지)
        if cx < 300:
            SavePath = SavePath + 'Back\\' + fileName
            #print("back")
            #print("cx, imgX - ", cx, imgX)
        else:
            SaveFlag = False
        '''
    if SaveFlag:
        cv.imwrite(SavePath, finalImg)
    return  finalImg


def Execute(filePath):
    fileName = path.split(filePath)[1]

    # 배경 제거된 이미지 로드
    grabedImgPath = filePath
    grabedImg = cv.imread(grabedImgPath)

    #원본 이미지 로드
    srcFolderPath = 'E:\\Study\\Pill\\Firstdis_images\\image\\Class5\\Org\\'
    orgImgPath = srcFolderPath + fileName
    orgImg = cv.imread(orgImgPath)
    '''
    cv.imshow('img', grabedImg)
    cv.waitKey(0)
    cv.imshow('orgImg', orgImg)
    cv.waitKey(0)
        '''
    contours, lasIdx = PositionManager.GetPillContour(grabedImg, 200, True)
    maxArea = 0
    idx = 0
    numberOfContour= len(contours)
    if numberOfContour == 1:
        tmp = CropImg(grabedImg, orgImg, contours[0], fileName, False)

    elif numberOfContour > 2:
        print('Contour Number : ', numberOfContour)
        print(fileName)
        dst = 'E:\\Study\\Pill\\Firstdis_images\\image\\Circle\\re\\' + fileName
        #copyfile(each, dst)
        '''
        for cnt in contours:
            area = cv.contourArea(cnt)
            tmp = CropImg(grabedImg, orgImg, cnt, fileName, False)
        '''
    elif numberOfContour == 2:
        croppingImgs = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if maxArea < area:
                maxIdx = idx
                maxArea = area
            #print(area)
            tmp = CropImg(grabedImg, orgImg, cnt, fileName, False)
            croppingImgs.append(tmp)
            # cv.drawContours(img, contours, idx, (0, 255, 0), 1)
            idx = idx + 1


'''
진입 점
'''
import os.path as path
import glob

grabedFolderPath = 'E:\\Study\\Pill\\ForegroundTest\\FirstDis\\fcn'
FilePathList = glob.glob(path.join(grabedFolderPath, '*.*'))
for each in FilePathList:
    print(each)
    Execute(each)
    #os.remove(each)








