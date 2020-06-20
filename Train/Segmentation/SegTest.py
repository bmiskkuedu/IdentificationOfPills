
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.unet import resnet50_unet

import cv2 as cv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# 모델 입력 이미지 사이즈는 32의 배수가 되어야 한다.
ModelInputHeight = 640
ModelInputWidth = 640
Model_Path = "E:\\SKKU\\PillRecog\\PillRecogApp\\Engine\\Models\\Background\\background.hdf4"

model = vgg_unet(n_classes=2 ,  input_height=ModelInputHeight, input_width=ModelInputWidth  )
#model = resnet50_unet(n_classes=2 ,  input_height=ModelInputHeight, input_width= ModelInputWidth )

model.load_weights(Model_Path)

OneDataFlag = True

# 테스트로 1개 파일에 대해 적용
if OneDataFlag :
    FilePath = 'E:\\198200157.jpg'
    #FilePath = 'E:\\2.jpg'
    out = model.predict_segmentation(
        inp=FilePath,
        out_fname="E:\\Study\\Pill\\SegTest\\Data\\dd\\out.png"
    )

    org = cv.imread(FilePath)
    # 모델 아웃풋 사이즈는 입력 사이즈의 절반이다.
    org = cv.resize(org, (ModelInputHeight//2, ModelInputWidth//2))

    out2 = np.array(out * 1, dtype = np.uint8)

    # 원본 데이터와 배열 차원 수를 맞춰주기 위해 BGR로 변경 한다.
    out2 = cv.cvtColor(out2, cv.COLOR_GRAY2BGR)

    # 원본 데이터와 mask 데이터의 모양 체크
    print(org.shape)
    print(out2.shape)
    #cv.imshow("FilePath", org)
    #cv.waitKey(0)

    # 원본 데이터에 mask데이터를 곱한다.
    # mask 데이터는 배경은 0, 전경은 1로 구성되어 있다.
    # 따라서 원본 데이터의 배경은 0이 된다.
    masked = org * out2
    cv.imshow("masked", masked)
    cv.waitKey(0)
    #cv.imwrite(FilePath, masked)

    import matplotlib.pyplot as plt
    plt.imshow(out)
    plt.show()

else :
    import os.path as path
    import glob

    DataParentPath = 'E:\\Study\\Pill\\ForegroundTest\\FirstDis\\fcn'

    TargetPath = DataParentPath
    FilePathList = glob.glob(path.join(TargetPath, '*.*'))

    for each in FilePathList:
        print(each)
        FilePath = each
        out = model.predict_segmentation(
            inp=FilePath,
            out_fname="E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\out.png"
        )

        org = cv.imread(FilePath)
        org = cv.resize(org, (ModelInputHeight//2, ModelInputWidth//2))

        out2 = np.array(out * 1, dtype=np.uint8)

        out2 = cv.cvtColor(out2, cv.COLOR_GRAY2BGR)
        #cv.imshow("threshed", threshed)
        #cv.waitKey(0)
        print(org.shape)
        print(out2.shape)
        masked = org * out2
        #cv.imshow("masked", masked)
        #cv.waitKey(0)
        cv.imwrite(each, masked)