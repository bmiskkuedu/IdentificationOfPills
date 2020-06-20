from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model

import cv2
from Imprint import DictionaryOfImprintClass


# key : 폴더이름, value : class

def CalcDescentOrdering(predList):
    maxValue = 0
    maxValueIdx = 0

    tmp = 0.0
    clonePredList = predList.copy()
    listLen = len(clonePredList)
    for i in range(0, listLen - 1):
        for j in range(i, listLen):
            #print(clonePredList[i], clonePredList[j])
            if clonePredList[i] < clonePredList[j]:
                tmp = clonePredList[i]
                clonePredList[i] = clonePredList[j]
                clonePredList[j] = tmp

    # top1의 idx부터 시작하는 List
    topDownidxList = []
    for each in clonePredList:
        idx = 0
        for each2 in predList:
            if each == each2:
                topDownidxList.append(idx)
            idx = idx + 1

    return topDownidxList


from Utils import CroppingForeground


def PredictTheTarget(filePath, model):

    isCapsuleType = False
    #loadedImag = ForegroundPreProcess2.ExcuteCroping(FilePath=filePath, ShowFlag=False, isCapsuleType = isCapsuleType, hist=10, FileSaveFlag=False)
    loadedImag = CroppingForeground.ExcuteCroping(FilePath=filePath, ShowFlag = False, SaveFlag = False, Blur = True)
    loadedImag = cv2.resize(loadedImag,(75,75))
    loadedImag = cv2.cvtColor(loadedImag, cv2.COLOR_GRAY2BGR)


    cv2.imshow('loadedImag', loadedImag)
    cv2.waitKey(0)
    if isCapsuleType:
        loadedImag = cv2.cvtColor(loadedImag, cv2.COLOR_BGR2RGB)

    convertToArray = img_to_array(loadedImag)
    scaled = convertToArray / 255.0

    predTmp = model.predict(scaled[np.newaxis, ...])
    print(predTmp)
    #y_classes = keras.np_utils.probas_to_classes(predTmp)
    descentOrderList = CalcDescentOrdering(predTmp[0])
    print(descentOrderList)
    
    # Top10 출력    
    idx = 0
    classDic = DictionaryOfImprintClass.GetClass1_2ColorClass0()
    #classDic = TestClassDic.GetClass3()

    for each in descentOrderList:
        for key, value in classDic.items():
            if value == each:
                # 순위, 폴더명
                print(idx, key)
        idx = idx + 1
        if idx > 10:
            break;


if __name__ == "__main__" :

    import os.path as path
    import glob


    DataParentPath = "E:\\Study\\Pill\\ForegroundTest\\FirstDis\\Class1_2\\"
    Model_Path = DataParentPath + 'Model\\Class0\\1417-0.9999.hdf5'
    #Model_Path = DataParentPath + 'Model\\228-0.9914.hdf5'
    #Model_Path = 'E:\\SKKU\\PillRecog\\PillRecogApp\\Engine\\Models\\Imprint\\Class0_Color0.hdf5'


    model = load_model(Model_Path)

    TargetPath = DataParentPath + 'Test\\Class0'
    #TargetPath = DataParentPath + 'Test'
    FilePathList = glob.iglob(TargetPath + '/**/*.jpg', recursive=True)


    for each in FilePathList:
        print(each)
        PredictTheTarget(each, model)




