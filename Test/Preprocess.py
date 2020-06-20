'''
입력된 이미지에 전처리를 진행하여 반환한다.
전처리 내용 : 입력 이미지에서 전경 비율이 높도록 배경을 잘라낸다.
'''
import PositionManager
import cv2 as cv
import Configuration

def ForegroundCrop(FilePath, ShowFlag = False):
    '''
    입력 이미지의 전경의 위치를 이용하여
    적절한 전경 상하좌우의 위치를 반환한다.
    :param FilePath:
    :param ShowFlag:
    :return:
    '''
    try:

        imgY = PositionManager.foreGroundImg.shape[0]
        imgX = PositionManager.foreGroundImg.shape[1]

        width = PositionManager.maxX - PositionManager.minX
        height = PositionManager.maxY - PositionManager.minY

        diffValue = abs(width - height)

        # 정사각형 배경 안에 전경 이미지 영역을 최대한 크게 한다.
        offsetValue = 10
        if width > height:
            xOffset = offsetValue
            yOffset = diffValue // 2 + offsetValue
        elif width < height:
            xOffset = diffValue // 2 + offsetValue
            yOffset = offsetValue
        else:
            xOffset = offsetValue
            yOffset = offsetValue

        initX = PositionManager.minX - xOffset
        if initX < 0:
            initX = 0

        maxX = PositionManager.maxX + xOffset

        if maxX > imgX:
            maxX = imgX

        initY = PositionManager.minY - yOffset

        if initY < 0:
            initY = 0

        maxY = PositionManager.maxY + yOffset
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

        if ShowFlag:
            print(initX, maxX, initY, maxY)
            print(imgX, imgY)

       # img_gray = cv.cvtColor(PositionManager.foreGroundImg, cv.COLOR_BGR2GRAY)
        img = PositionManager.foreGroundImg
    except Exception:
        '''
        img = cv.imread(FilePath)
        totalHight = math.ceil(img.shape[0] * 0.9)
        totalWidth = math.ceil(img.shape[1] * 0.9)
        croppingImg = img[0:totalHight, 0:totalWidth]
        cv.imwrite(FilePath, croppingImg)
        ForegroundCrop(FilePath, ShowFlag=False)
        '''
    return  img, initX, maxX, initY, maxY


