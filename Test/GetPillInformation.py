'''
입력된 이미지의 모양을 결정한다.
입력된 이미지를 Imprint 인식 모델의 입력 이미지로 사용할
이미지로 변환한다.
'''
import PositionManager
import ColorManager
import cv2 as cv
import Configuration
from keras.preprocessing.image import load_img, img_to_array
import Preprocess
import numpy as np
from keras.models import load_model

def GetPillShapeColorImprint(FilePath, ShowFlag = False):
    '''
    알약의 모양과 Imprint를 인식하기 위한 이미지를 반환한다.
    :param FilePath:
    :param ShowFlag:
    :return:
    '''
    classDic = {'Class0': 0, 'Class1': 1, 'Class10': 2, 'Class11': 3, 'Class12': 4, 'Class13': 5, 'Class14': 6,
                'Class2': 7, 'Class3': 8, 'Class4': 9, 'Class5': 10, 'Class6': 11, 'Class7': 12, 'Class8': 13,
                'Class9': 14}

    # 알약의 위치 좌표 정보를 계산한다.
    PositionManager.CalcPillPositionInfo(FilePath, ShowFlag)
    img, initX, maxX, initY, maxY = Preprocess.ForegroundCrop(FilePath, ShowFlag)

    # 알약의 전경 영역을 가져온다.
    maskimg = PositionManager.ImgMasking

    # 불필요한 배경 영역을 잘라낸다
    croppingImg = maskimg[initY:maxY, initX:maxX]

    # 미리 정해진 사이즈로 변경한다.
    croppingImg = cv.resize(croppingImg, (Configuration.ImageSize, Configuration.ImageSize))
    ret, bin = cv.threshold(croppingImg, 30, 255, cv.THRESH_BINARY)

    if ShowFlag:
        cv.imshow('bin', bin)
        cv.waitKey(0)

    model_Path ='Models/Shape/ShapeDetectModel.hdf5'

    model = load_model(model_Path)

    convertToArray = img_to_array(bin)
    scaled = convertToArray / 255.0

    # 결과값이 확률값으로 출력됨
    predTmp = model.predict(scaled[np.newaxis, ...])
    #print(predTmp)

    #결과 값중 가장 큰 값을 계산한다.
    idx = 0
    maxValue = 0
    maxValueIdx = 0
    for each in predTmp[0]:
        if each > maxValue:
            maxValue = each
            maxValueIdx = idx

        idx = idx + 1

    # 가장 큰 값의 class이름을 가져온다.
    keys = list(classDic.keys())
    className = keys[maxValueIdx]

    pillColor, sumedLowRatioValue, colors = ColorManager.GetColorInfo(forgroundImg=img, kClustterValue=5,
                                                                      ShowFlag=False)
    # Imprint 이미지를 구한다.
    imprintImg = GetImprintImg(img, className, initX, maxX, initY, maxY)

    return className, imprintImg, pillColor

def GetImprintImg(img, className, initX, maxX, initY, maxY, ShowFlag = False):
    '''
    Imprint를 인식할 이미지를 만들어 반환한다.

    :param img:
    :param className:
    :param initX:
    :param maxX:
    :param initY:
    :param maxY:
    :param ShowFlag:
    :return:
    '''

    isCapsuleType = False
    if className == 'Class3':
        isCapsuleType = True
    if isCapsuleType == False:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=Configuration.HistEqValue, tileGridSize=(5, 5))
        img = clahe.apply(img_gray)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    if ShowFlag:
        cv.imshow("img_gray", img)
        cv.waitKey(0)

        # 테두리 주변에 노이즈가 있을 수 있으므로
        # 테두리 주변을 일정한 두께로 검정색 마스킹을 수행
    cv.drawContours(img, PositionManager.contours, PositionManager.lastIdx, (0, 0, 0), 5)

    # 전경 비율이 크게 나오도록 배경을 잘라냄
    croppingImg = img[initY:maxY, initX:maxX]

    # 이미지 크기를 정해진 크기로 변경한다.
    croppingImg = cv.resize(croppingImg, (Configuration.ImageSize, Configuration.ImageSize))

    return  croppingImg