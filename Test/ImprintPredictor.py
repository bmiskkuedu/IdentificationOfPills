from keras.preprocessing.image import load_img, img_to_array
import glob
import os.path as path
import numpy as np
from keras.models import load_model
import Configuration
import cv2

import ImprintPredictDictionary

def CalcDescentOrdering(predList):
    '''
    predList의 값을 오름차순 변경하여 반환한다.
    :param predList:
    :return:
    '''
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

import ImprintPredictDictionary
def GetImprintModelInfo(shape,  color) :

    modelName = ''
    testClassDic = ''
    if shape == Configuration.Shape.Circle :
        if color ==  Configuration.Color.Class0:
            modelName ='Class0_Color0.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass0ColorClass0()
        elif color == Configuration.Color.Class1 or color == Configuration.Color.Class6 or color == Configuration.Color.Class7:
            modelName = 'Class0_Color1_6_7.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass0ColorClass1_6_7()
        elif color == Configuration.Color.Class2:
            modelName = 'Class0_Color2.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass0ColorClass2()
        elif color == Configuration.Color.Class3:
            modelName = 'Class0_Color3.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass0ColorClass3()
        elif color == Configuration.Color.Class4:
            modelName = 'Class0_Color4.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass0ColorClass4()
        elif color == Configuration.Color.Class5:
            modelName = 'Class0_Color5.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass0ColorClass5()
    elif shape == Configuration.Shape.Ellipse or  shape == Configuration.Shape.Oblong :
        if color ==  Configuration.Color.Class0:
            modelName ='Class1_2_Color0.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass1_2ColorClass0()
        elif color ==  Configuration.Color.Class1:
            modelName = 'Class1_2_Color1.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass1_2ColorClass1()
        elif color == Configuration.Color.Class2:
            modelName = 'Class1_2_Color2.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass1_2ColorClass2()
        elif color ==  Configuration.Color.Class3:
            modelName = 'Class1_2_Color3.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass1_2ColorClass3()
        elif color ==  Configuration.Color.Class4:
            modelName = 'Class1_2_Color4.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass1_2ColorClass4()
        elif color ==  Configuration.Color.Class5 or color ==  Configuration.Color.Class7:
            modelName = 'Class1_2_Color5_7.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass1_2ColorClass5_7()
        elif color ==  Configuration.Color.Class6:
            modelName = 'Class1_2_Color6.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass1_2ColorClass6()
    elif shape == Configuration.Shape.Capsule:
        modelName = 'Class3.hdf5'
        testClassDic = ImprintPredictDictionary.GetClass3()
    elif shape == Configuration.Shape.Triangle:
        modelName = 'Class4.hdf5'
        testClassDic = ImprintPredictDictionary.GetClass4()
    elif shape == Configuration.Shape.Rectangular:
        modelName = 'Class5.hdf5'
        testClassDic = ImprintPredictDictionary.GetClass5()
    elif shape == Configuration.Shape.Rhombus or shape == Configuration.Shape.Pentagon:
        modelName = 'Class6_7.hdf5'
        testClassDic = ImprintPredictDictionary.GetClass6_7()
    elif shape == Configuration.Shape.Hexagon:
        modelName = 'Class8.hdf5'
        testClassDic = ImprintPredictDictionary.GetClass8()
    elif shape == Configuration.Shape.Octagon or shape == Configuration.Shape.Peanut or shape == Configuration.Shape.ETCShape:
        if color == Configuration.Color.Class0:
            modelName = 'Class9_10_14_Color0.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass9_10_14ColorClass0()
        elif color == Configuration.Color.Class1 or color == Configuration.Color.Class2 or color == Configuration.Color.Class3:
            modelName = 'Class9_10_14_Color1_2_3.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass9_10_14ColorClass1_2_3()
        elif color == Configuration.Color.Class4 or color == Configuration.Color.Class5 or color == Configuration.Color.Class6\
                or color == Configuration.Color.Class7:
            modelName = 'Class9_10_14_Color4_5_6_7.hdf5'
            testClassDic = ImprintPredictDictionary.GetClass9_10_14ColorClass4_5_6_7()
    elif shape == Configuration.Shape.WaterDrop:
        modelName = 'Class11.hdf5'
        testClassDic = ImprintPredictDictionary.GetClass11()
    elif shape == Configuration.Shape.Heart:
        modelName = 'Class12.hdf5'
        testClassDic = ImprintPredictDictionary.GetClass12()
    elif shape == Configuration.Shape.Shield:
        modelName = 'Class13.hdf5'
        testClassDic = ImprintPredictDictionary.GetClass13()
    return modelName, testClassDic

def ChangeShapeInfo(shapeClassName):
    shapeRetValue = Configuration.Shape.ETC

    if shapeClassName == 'Class0':
        shapeRetValue = Configuration.Shape.Circle
    elif shapeClassName == 'Class1':
        shapeRetValue = Configuration.Shape.Ellipse
    elif shapeClassName == 'Class2':
        shapeRetValue = Configuration.Shape.Oblong
    elif shapeClassName == 'Class3':
        shapeRetValue = Configuration.Shape.Capsule
    elif shapeClassName == 'Class4':
        shapeRetValue = Configuration.Shape.Triangle
    elif shapeClassName == 'Class5':
        shapeRetValue = Configuration.Shape.Rectangular
    elif shapeClassName == 'Class6':
        shapeRetValue = Configuration.Shape.Rhombus
    elif shapeClassName == 'Class7':
        shapeRetValue = Configuration.Shape.Pentagon
    elif shapeClassName == 'Class8':
        shapeRetValue = Configuration.Shape.Hexagon
    elif shapeClassName == 'Class9':
        shapeRetValue = Configuration.Shape.Octagon
    elif shapeClassName == 'Class10':
        shapeRetValue = Configuration.Shape.Pentagon
    elif shapeClassName == 'Class11':
        shapeRetValue = Configuration.Shape.WaterDrop
    elif shapeClassName == 'Class12':
        shapeRetValue = Configuration.Shape.Heart
    elif shapeClassName == 'Class13':
        shapeRetValue = Configuration.Shape.Shield
    elif shapeClassName == 'Class14':
        shapeRetValue = Configuration.Shape.ETCShape
    else :
        shapeRetValue = Configuration.Shape.ETC

    return  shapeRetValue


def PredictImprint(img, shapeClassInfo, colorClassname):
    shapeClassName = ChangeShapeInfo(shapeClassInfo)
    modelName, testClassDic = GetImprintModelInfo(shapeClassName, colorClassname)
    modelPath = 'Models/Imprint/' + modelName
    model = load_model(modelPath)
    print(modelPath)
    convertToArray = img_to_array(img)
    scaled = convertToArray / 255.0

    predTmp = model.predict(scaled[np.newaxis, ...])
    #print(predTmp)
    # y_classes = keras.np_utils.probas_to_classes(predTmp)
    descentOrderList = CalcDescentOrdering(predTmp[0])
    #print(descentOrderList)

    # Top10 출력
    idx = 0
    classDic = testClassDic

    folderList = []
    for each in descentOrderList:
        for key, value in classDic.items():
            if value == each:
                # 순위, 폴더명
                #print(idx, key)
                folderList.append(key)
        idx = idx + 1
        if idx > 20:
            break;
    return folderList