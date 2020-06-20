
from keras.preprocessing.image import img_to_array

import numpy as np
from keras.models import load_model
from Shape import ShapePreprocess

classDic = {'Class0': 0, 'Class1': 1, 'Class10': 2, 'Class11': 3, 'Class12': 4, 'Class13': 5, 'Class14': 6, 'Class2': 7, 'Class3': 8, 'Class4': 9, 'Class5': 10, 'Class6': 11, 'Class7': 12, 'Class8': 13, 'Class9': 14}

def PredictTheTarget(FilePath, model, ShowFlag = False):
    bin = ShapePreprocess.excute(FilePath, ShowFlag, False)
    #gray = cv.imread(FilePath)
    #gray = cv.resize(gray, (150,150))
    convertToArray = img_to_array(bin)
    scaled = convertToArray / 255.0

    # 결과값이 확률값으로 출력됨
    predTmp = model.predict(scaled[np.newaxis, ...])
    print(predTmp)
    idx = 0

    maxValue = 0
    maxValueIdx = 0
    for each in predTmp[0]:
        if each > maxValue:
            maxValue = each
            maxValueIdx = idx

        idx = idx + 1

    keys = list(classDic.keys())
    className = keys[maxValueIdx]
   
    print(className)

    return



if __name__ == "__main__" :

    import os.path as path
    import glob

    DataParentPath = "E:\\Study\\Pill\\TestData\\FullTest"
    #DataParentPath = "E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\grab\\"

    TargetPath = DataParentPath
   #FilePathList = glob.glob(path.join(TargetPath, '*.jpg'))
    FilePathList = glob.iglob(TargetPath + '/**/*.jpg', recursive=True)
    #print(FilePathList)
    #ParentPath = "E:\\Study\\Pill\\Shape\\GrayScale\\"
    #Model_Path = ParentPath + '\\Model\\50-0.9914.hdf5'
    Model_Path = 'E:\\Study\\Pill\\Shape\\Binary\\Model\\62-0.9918.hdf5'
    model = load_model(Model_Path)
    #model2 = load_model(Model_Path)
    #oneTest = DataParentPath + '\\52.jpg'
    #PredictTheTarget(oneTest, model)

    #FilePathList = []

    #FilePathList.append('E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\grab\\230.jpg')
    #FilePathList.append('E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\grab\\251.jpg')
    #FilePathList.append('E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\grab\\170.jpg')
    #FilePathList.append('E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\grab\\171.jpg')

    for each in FilePathList:
        print(each)

        PredictTheTarget(each, model, False )
