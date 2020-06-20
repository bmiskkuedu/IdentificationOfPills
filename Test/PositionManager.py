'''
알약 전경 위치 정보 반환 작업 수행
 - 외곽선 좌표 반환
 - 중심 좌표 반환
 - 배경이 삭제된 전경 이미지 반환
     - 딥러닝으로 배경이 삭제된 이미지는 Imprint의 훼손이 있을수 있다.
     - 딥러닝으로 배경이 삭제된 이미지의 전경 좌표를 이용해서
        원본 이미지에서 해당 좌표의 이미지만 살리고 그 외는 검정색으로 마스킹 처리한다.
'''

import cv2 as cv
import numpy as np
import Configuration

def GetYValueListInContour(contour, xPos):
    '''
    xPos에 해당하는 y좌표를 contour에서 찾아 반환한다.
    '''
    yList = []
    for each in contour:
        currentXPos = each[0][0]
        currentYPos = each[0][1]
        if currentXPos == xPos:
            yList.append(currentYPos)

    return yList

def GetMinMaxPosInContour(contour, maskedImg):
    global minX,minY,maxX,maxY
    '''
    contour의 x,y 좌표에서 최대 최소, 중심 좌표를 반환한다.
    :param contour:
    :param cx:
    :param cy:
    :return:
    '''

    # 중심점을 구한다.
    M = cv.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    '''
    cv.circle(maskedImg, (cx, cy), 10, (0, 255, 255), -1)
    cv.imshow("maskedImg", maskedImg)
    cv.waitKey(0)
    '''
    # 초기값으로 외곽선의 중심 좌표를 넣어 준다.
    minX,minY,maxX,maxY = cx, cy, cx, cy

    for each in contour:
        currentX = each[0][0]
        currentY = each[0][1]
        if currentX < minX:
            minX = currentX
        elif currentX > maxX:
            maxX = currentX

        if currentY < minY:
            minY = currentY
        elif currentY > maxY:
            maxY = currentY

    return minX,minY,maxX,maxY, cx, cy

def GetPillContour(img_color, highThreshold = 50, showFlag = False):
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    img_gray = cv.medianBlur(img_gray, 5)
    if showFlag:
        cv.imshow("Blur", img_gray)
        cv.waitKey(0)

    img_colorTmp = img_color.copy()

    low = 0
    img_canny = cv.Canny(img_gray, low, highThreshold)
    if showFlag:
        cv.imshow("img_canny", img_canny)
        cv.waitKey(0)

    adaptive = cv.adaptiveThreshold(img_canny, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 0)

    # contour에서 구멍 난 경우를 대비
    kernel = np.ones((10, 10), np.uint8)
    closing = cv.morphologyEx(adaptive, cv.MORPH_CLOSE, kernel)
    contourInput = closing
    contours, hierarchy = cv.findContours(contourInput, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    maxIdx = 0
    idx = 0
    maxArea = 0.0

    for cnt in contours:
        area = cv.contourArea(cnt)
        if maxArea < area:
            maxIdx = idx
            maxArea = area
        idx = idx + 1

    cv.drawContours(img_colorTmp, contours, idx - 1, (0, 0, 255), 1)
    if showFlag:
        cv.imshow("img_colorTmp", img_colorTmp)
        cv.waitKey(0)

        # 알약 외곽선을 추출한다.
    #contour = contours[idx - 1]
    lastIdx = idx - 1
    return contours, maxIdx#lastIdx

def hangulFilePathImageRead ( filePath ) :
    '''
    filePath의 경로에 한글이 있을 경우
    :param filePath:
    :return:
    '''

    stream = open( filePath.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)

    return cv.imdecode(numpyArray , cv.IMREAD_UNCHANGED)

from keras_segmentation.models.unet import resnet50_unet
def GetForegroundImgContourInfo(FilePath):
    '''
    딥러닝 모델을 이용하여 배경을 제거한다.
    전경의 외곽선 정보를 리턴한다.
    :param FilePath:
    :return:
    '''
    # 모델 입력 이미지 사이즈는 32의 배수가 되어야 한다.
    ModelInputHeight = 640
    ModelInputWidth = 640
    Model_Path = "Models\\Background\\background.hdf4"

    model = resnet50_unet(n_classes=2, input_height=ModelInputHeight, input_width=ModelInputWidth)

    model.load_weights(Model_Path)

    out = model.predict_segmentation(
        inp=FilePath,
        out_fname="out.png"
    )

    org = cv.imread(FilePath)
    # 모델 아웃풋 사이즈는 입력 사이즈의 절반이다.
    org = cv.resize(org, (320, 320))

    out2 = np.array(out * 1, dtype=np.uint8)
    # threshed = cv.adaptiveThreshold(out2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)
    out2 = cv.cvtColor(out2, cv.COLOR_GRAY2BGR)

    #print(org.shape)
    #print(out2.shape)
    # cv.imshow("FilePath", org)
    # cv.waitKey(0)

    masked = org * out2
    #cv.imshow("masked", masked)
    #cv.waitKey(0)
    maskedImg = cv.resize(masked, (Configuration.ImageSizeForPreprocess, Configuration.ImageSizeForPreprocess))

    # 알약 외곽선을 추출한다.
    highThreshold = 100
    contours, lastIdx = GetPillContour(img_color =maskedImg, highThreshold = highThreshold, showFlag = False)

    if contours is None:
        print("contour is not found!!")

    return contours, lastIdx


def GetForegroundImgInfo(FilePath, showFlag = False):
    '''
    원본 이미지에 딥러닝을 이용하여 1차적인 배경 삭제 작업을 진행한다.(Imprint 훼손 가능성 있음)
    딥러닝으로 배경 삭제 작업된 이미지의 전경 외곽선을 이용하여 다시 원본 이미지에서
    최종적인 전경 추출 이미지를 구한다.(Imprint 훼손 없이 배경 삭제 이미지 획득 목표)
    :param FilePath:
    :param showFlag:
    :return:
    '''
    #import time
    #start = time.time()
    #print(FilePath)
    img_color = cv.imread(FilePath)
    #cv.imshow('ss', img_color)
    #cv.waitKey(0)
    #img_color = hangulFilePathImageRead(FilePath)
    img_color = cv.resize(img_color, (Configuration.ImageSizeForPreprocess,Configuration.ImageSizeForPreprocess))

    contours, lastIdx = GetForegroundImgContourInfo(FilePath)

    if contours is None:
        print("contour is not found!!")
        
    # 검정색 배경을 만듦
    ImgMasking = np.zeros((img_color.shape[0], img_color.shape[1], 3), np.uint8)

    minX,minY,maxX,maxY, cx, cy = GetMinMaxPosInContour(contours[lastIdx], ImgMasking)

    '''
    중심점을 시각적으로 확인
    cv.circle(ImgMasking, (cx, cy), 10, (0, 255, 255), -1)
    cv.imshow("center point", ImgMasking)
    cv.waitKey(0)
    '''
    # 배경이 제거된 이미지에서 찾은 컨투어(외곽선)의 좌표를 위에서 만든 검정 배경에 그림
    cv.drawContours(ImgMasking, contours, lastIdx, (255, 255, 255), 0)
    if showFlag:
        cv.imshow("ImgMasking1", ImgMasking)
        cv.waitKey(0)

    # 내부를 흰색으로 채워 줌
    mask = np.zeros((img_color.shape[0] + 2, img_color.shape[1] + 2), np.uint8)
    mask[:] = 0
    cv.floodFill(ImgMasking, mask, (cx, cy), (255, 255, 255))
    #print(cx,cy)

    # 원본 이미지에 위에서 만든 mask를 씌워 원본 이미지의 배경을 제거 함.
    dst = cv.copyTo(img_color, ImgMasking)

    #print("time :", time.time() - start)
    if showFlag:
        cv.imshow("ImgMasking2", ImgMasking)
        cv.imshow("Last", dst)
        cv.waitKey(0)

    return dst, ImgMasking, contours, lastIdx, cx, cy, minX,minY,maxX,maxY

def IsItOutside(x1, y1, x2, y2):
    '''
    좌표가 알약 외부에 있는지 확인한다.
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return: 알약 외부에 있으면 True
    '''
    retValue = False
    if (minX > x1) or (maxX < x1):
        retValue = True
    elif (minX > x2) or (maxX < x2):
        retValue = True
    elif (minY > y1) or (maxY < y1):
        retValue = True
    elif (minY > y2) or (maxY < y2):
        retValue = True

    return  retValue

def GetSpliteLineInfo(foregroundImg, contours, lastIdx, minX,minY,maxX,maxY, showFlag = False):
    '''
    알약의 분할선을 인식
     - 최종적으로 미사용 함.
    :param foregroundImg:
    :param contours:
    :param lastIdx:
    :param minX:
    :param minY:
    :param maxX:
    :param maxY:
    :param showFlag:
    :return:
    '''
    img_gray = cv.cvtColor(foregroundImg, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=100, tileGridSize=(3, 3))
    img_gray = clahe.apply(img_gray)
    if showFlag :
        cv.imshow('img_gray', img_gray)
    # 노이즈 제거
    img_gray = cv.fastNlMeansDenoising(img_gray, None, 100, 7, 9)


    # 이진화, 알약 테두리 삭제
    img_binary = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 19, 10)

    if showFlag:
        cv.imshow('img_binary', img_binary)

    cv.drawContours(img_binary, contours, lastIdx, (0, 0, 0), 15)

    # 구멍난 부분 보수
    kernel = np.ones((10, 10), np.uint8)
    closing = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel)

    if showFlag:
        cv.imshow('closing', closing)

    result = closing

    pillHeight = maxY - minY
    pillWidth = maxX - minX

    # 직선 검출을 알약의 높이의 50% 이상의 길이로만 한정
    basicLength = round(pillHeight * 0.5)

    # 직선 검출 세부 설정 항목
    minLineLength = basicLength
    maxLineGap = 5
    threshold = 50

    # 직선 라인 검출
    lines = cv.HoughLinesP(result, rho=1, theta=1 * np.pi / 180, threshold=threshold, minLineLength=minLineLength,
                           maxLineGap=maxLineGap)
    lineImg = foregroundImg.copy()

    # 가로 세로 직선이 수직 수평이 아닐 경우 필터링 용도
    offsetLineValue = 10

    detectHorizontalLineFlag = False
    detectVerticallLineFlag = False

    minXOfSplitLine = maxX
    maxXOfSplitLine = minX
    minYOfSplitLine = maxY
    maxYOfSplitLine = minY

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if IsItOutside( x1, y1, x2, y2 ):
                continue

            # 가로 분할선 일 때
            if abs(x2 - x1) > basicLength:
                if abs(y2 - y1) < (pillHeight //offsetLineValue) :
                    detectHorizontalLineFlag = True
                    # y 축 좌표 정보를 저장한다.
                    if  minYOfSplitLine > y1:
                        minYOfSplitLine = y1

                    if maxYOfSplitLine < y2:
                        maxYOfSplitLine = y2

                    if showFlag:
                        cv.line(lineImg, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv.imshow('line', lineImg)
            # 세로 분할선 일 경우
            elif abs(y2 - y1) > basicLength:
                if abs(x2 - x1) < (pillWidth // offsetLineValue):
                    detectVerticallLineFlag = True
                    # x축 좌표 위치를 저장한다.
                    if minXOfSplitLine > x1:
                        minXOfSplitLine = x1

                    if maxXOfSplitLine < x2:
                        maxXOfSplitLine = x2

                    if showFlag:
                        cv.line(lineImg, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv.imshow('line', lineImg)

    lineInfo = Configuration.SplitLineKind.NoLines
    if detectHorizontalLineFlag:
        if detectVerticallLineFlag:
            lineInfo = Configuration.SplitLineKind.Cross
        else:
            lineInfo = Configuration.SplitLineKind.Horizontal
            # x축 좌표 정보를 0으로 해준다.
            # - y축 정보만 이용하여 글자 추출시 분할선을 마스킹 한다.
            minXOfSplitLine = 0
            maxXOfSplitLine = 0
    elif detectVerticallLineFlag:
        if detectHorizontalLineFlag:
            lineInfo = Configuration.SplitLineKind.Cross
        else:
            lineInfo = Configuration.SplitLineKind.Vertical
            # y축 좌표 정보를 0으로 해준다.
            # - x축 정보만 이용하여 글자 추출시 분할선을 마스킹 한다.
            minYOfSplitLine = 0
            maxYOfSplitLine = 0

    return lineInfo, minXOfSplitLine, maxXOfSplitLine, minYOfSplitLine, maxYOfSplitLine

def CalcPillPositionInfo(FilePath, showFlag = False):
    '''
    PositionManager의 최 상위 API
    :param FilePath:
    :param showFlag:
    :return:
    '''
    global foreGroundImg, ImgMasking, contours, lastIdx, cx, cy, minX, minY, maxX, maxY, splitLineInfo, minXOfSplitLine, maxXOfSplitLine, minYOfSplitLine, maxYOfSplitLine

    foreGroundImg, ImgMasking, contours, lastIdx, cx, cy, minX, minY, maxX, maxY = GetForegroundImgInfo(
        FilePath, showFlag)
    splitLineInfo, minXOfSplitLine, maxXOfSplitLine, minYOfSplitLine, maxYOfSplitLine = GetSpliteLineInfo(
        foreGroundImg, contours, lastIdx, minX, minY, maxX, maxY, showFlag)


'''
Unit Test
'''
if __name__ == '__main__':
    import glob
    import os.path as path

    GetForegroundImgInfo('E:\\Study\\Pill\\SegTest\\Data\\Test\\7.jpg', True)
    GetForegroundImgContourInfo('E:\\Study\\Pill\\SegTest\\Data\\Test\\7.jpg')

    DataParentPath = 'E:\\Study\\AI\\data\\pill-recognition\\data\\mypill\\Capsule\\'
    fileName = '00056-0170-70_PART_1_OF_1_CHAL10_SF_95134AFA.jpg'
    fileName = '15.jpg'
    FilePath = DataParentPath + fileName
    FlagOfExcuteKind = True
    if FlagOfExcuteKind:
        dst, ImgMasking, contours, lastIdx, cx, cy, minX, minY, maxX, maxY = GetForegroundImgInfo(
            FilePath, False)
        ret, minXOfSplitLine, maxXOfSplitLine, minYOfSplitLine, maxYOfSplitLine = GetSpliteLineInfo(dst, contours, lastIdx, minX, minY, maxX, maxY, True)
        print(ret)
        cv.waitKey(0)

    else:

        TargetPath = DataParentPath
        FilePathList = glob.glob(path.join(TargetPath, '*.*'))

        for each in FilePathList:
            print(each)

            dst, ImgMasking, contours, lastIdx, cx, cy, minX, minY, maxX, maxY = GetForegroundImgInfo(
                each)
            ret, minXOfSplitLine, maxXOfSplitLine, minYOfSplitLine, maxYOfSplitLine = GetSpliteLineInfo(
                dst, contours, lastIdx, minX, minY, maxX, maxY)
            print(ret)

#savePath = DataParentPath + "\\mask\\" + fileName
   #cv.imwrite(savePath, dst)

