
import sys
if __name__ == "__main__" :
    FilePath = sys.argv[1]

    print(FilePath)
    #FilePath = 'E:\\test.jpg'

    import GetPillInformation
    import ImprintPredictor

    # 알약의 모양, 색상, 임프린트를 인식할 이미지를 얻어온다.
    shapeClassName, ImprintImg, colorClassName = GetPillInformation.GetPillShapeColorImprint(FilePath)
    print('Pill Recognization Result')
    print('Shape Class : ', shapeClassName)
    print(colorClassName)

    # 임프린트를 인식한 결과를 얻어온다.
    #  - 임프린트 인식 모델을 학습시 사용한 학습 데이터의 폴더 명으로, 예측 확률이 높은 상위 11개를 얻어온다.
    folderNameList = ImprintPredictor.PredictImprint(ImprintImg, shapeClassName, colorClassName)

    for each in folderNameList:
        print("Imprint : " + each)

    ''''''
    # cv.imshow('d', ImprintImg)
    # cv.waitKey(0)