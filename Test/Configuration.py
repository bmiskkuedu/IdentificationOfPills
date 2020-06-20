import enum

KClustterValue = 5
ContourSensitivity = 0.03

ImageSize = 150
ImageSizeForPreprocess = 300

HistEqValue = 10

class Color(enum.Enum):
    Class0 = 0
    Class1 = 1
    Class2 = 2
    Class3 = 3
    Class4 = 4
    Class5 = 5
    Class6 = 6
    Class7 = 7
    ETC = 99

class Shape(enum.Enum):
    Circle = 0
    Ellipse = 1
    Oblong = 2
    Capsule = 3
    Triangle = 4
    Rectangular = 5
    Rhombus = 6
    Pentagon = 7
    Hexagon = 8
    Octagon = 9
    Peanut = 10
    WaterDrop = 11
    Heart = 12
    Shield = 13
    ETCShape = 14
    ETC = 99

class SplitLineKind(enum.Enum):
    NoLines = 0
    Horizontal = 1
    Vertical = 2
    Cross = 3
    ETC =4

def GetPoligonShape(lineNumer):
    retValue = Shape.Circle

    if lineNumer == 3:
        retValue =  Shape.Triangle
    elif lineNumer == 4:
        retValue = Shape.Rectangular
    elif lineNumer == 5:
        retValue = Shape.Pentagon
    elif lineNumer == 6:
        retValue = Shape.Hexagon
    elif lineNumer > 6:
        retValue = Shape.Octagon

    return retValue


def GetColor(hsvInfo):
    hue = hsvInfo[0]
    saturation = hsvInfo[1]
    value = hsvInfo[2]

    retValue = Color.ETC

    if saturation < 20 and value > 150:
        retValue = Color.Class0
    elif value < 70:
        retValue = Color.Class1
    elif hue < 10 or hue > 160:
        retValue = Color.Class2
    elif hue <= 40:
        retValue = Color.Class3
    elif hue <= 70:
        retValue = Color.Class4
    elif hue <= 100:
        retValue = Color.Class5
    elif hue <= 130:
        retValue = Color.Class6
    elif hue <= 160:
        retValue = Color.Class7
   
    return retValue

def GetLowSaturationInfo(colorInfo):
    returnValue = 0

    if colorInfo == Color.White:
        returnValue = 170
    elif colorInfo == Color.ORANGE:
        returnValue = 70
    elif colorInfo == Color.BROWN:
        returnValue  = 25

    return  returnValue