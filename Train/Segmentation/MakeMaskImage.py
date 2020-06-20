# Mask Image를 만든다.
# 만드는 순서
# 원본 이미지 -> Binary 이미지 -> 전체 이미지 데이터를 255로 나눔
# -> Binary이미지의 흰색 부분은 1, 나머지는 0으로 구성되는 Mask 이미지 데이터 생성

import cv2 as cv
import numpy
import sys
import glob
numpy.set_printoptions(threshold=sys.maxsize)
import os.path as path
DataParentPath = "E:\\Study\\Pill\\SegTest\\Data\\Tmp\\mask\\aug"
'''
FilePath = 'E:\\Study\\Pill\\SegTest\\Data\\Mask\\aug\\196400046_0_2459.png'
aa = cv.imread(FilePath, cv.COLOR_BGR2GRAY)
print(aa)
DataParentPath = 'E:\\Study\\Pill\\SegTest\\Data\\Mask\\aug'
'''
TargetPath = DataParentPath
FilePathList = glob.glob(path.join(TargetPath, '*.*'))

IsMaskWork = True

# 원본 이미지에서 Mask 이미지 생성 시
if IsMaskWork :
    #TargetPath ="E:\\Study\\Pill\\SegTest\\Data\\Tmp\\org\\grab"
    TargetPath = "E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\Aug\\Mask\\"
    FilePathList = glob.glob(path.join(TargetPath, '*.jpg'))
    for each in FilePathList:
        print(each)
        desFilePath = path.splitext(each)[0] + ".png"
        #print(desFilePath)

        # 원본 이미지를
        aaa = cv.imread(each, cv.IMREAD_GRAYSCALE)
        # mask data는 0, 1 값으로만 구성됨(배경 : 0, 전경 :1 )
        ret, bin = cv.threshold(aaa, 20, 255, cv.THRESH_BINARY)
        # mask data는 0, 1 값으로만 구성됨(배경 : 0, 전경 :1 )
        bin = bin / 255
        #print(bin)
        #cv.imshow("bb", bin)
        #cv.waitKey(0)
        cv.imwrite(desFilePath, bin)
        #break
# Binary Image에서 Mask 이미지 생성 시
else :

    TargetPath = "E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\Training\\Mask\\"
    FilePathList = glob.glob(path.join(TargetPath, '*.png'))
    for each in FilePathList:
        print(each)

        aaa = cv.imread(each, cv.IMREAD_GRAYSCALE)

        bin = aaa / 255
        # print(bin)
        # cv.imshow("bb", bin)
        # cv.waitKey(0)
        cv.imwrite(each, bin)
