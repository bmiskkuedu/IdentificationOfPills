# 배경 이미지를 잘라내어 전체 이미지에서 배경 비율을 줄이고
# 전경 비율을 늘린다.

from Utils import CroppingForeground
from Utils import PositionManager
import cv2 as cv
import os.path as path
import glob

def excute(FilePath, ShowFlag, FileSaveFlag):
    img, initX, maxX, initY, maxY = CroppingForeground.ForegroundCrop(FilePath, ShowFlag)
    maskimg = PositionManager.ImgMasking
    #cv.drawContours(maskimg, PositionManager.contours, PositionManager.lastIdx, (0, 0, 0), 5)

    croppingImg = maskimg[initY:maxY, initX:maxX]
    croppingImg = cv.resize(croppingImg, (150, 150))
    ret,bin = cv.threshold(croppingImg, 30, 255, cv.THRESH_BINARY)
    if ShowFlag:
        cv.imshow('bin', bin)
        cv.waitKey(0)
    if FileSaveFlag:
        fileName =   path.split(FilePath)[1]
       # print(fileName)
        #savePath = 'E:\\Study\\Pill\\Shape\\Data\\c\\' + fileName
       # print(savePath)
        cv.imwrite(FilePath, bin)

    return bin

if __name__ == "__main__" :



    DataParentPath1 = "E:\\Study\\Pill\\Shape\\data"

    TargetPath = DataParentPath1
    # FilePathList = glob.glob(path.join(TargetPath, '*.*'))

    FilePathList = glob.iglob(TargetPath + '/**/*.jpg', recursive=True)
    for each in FilePathList:
       # each = 'E:\\Study\\Pill\\Shape\\Binary\\Train\\Class0\\ACEC100TAB0029.jpg'
        print(each)
        excute(FilePath=each, ShowFlag=False, FileSaveFlag=True)