from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os.path as path
import os
import glob


# we create two instances with the same arguments
data_gen_args = dict(rotation_range=10,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     brightness_range=[0.95,1.30],
                     horizontal_flip=False,
                     vertical_flip=False,
                     shear_range=0.1,
                     zoom_range=0.1,
                     )
train_datagen = ImageDataGenerator(**data_gen_args)

def MakeAugData(isTrainCase, currentFolderPath):

    UsedFolderPath = ''
    maxBasisNumber = 1000
    seedNumber = 1
    if isTrainCase is False:
        maxBasisNumber = 300
        seedNumber = 3

    filePathList = glob.iglob(currentFolderPath + '/**/*.jpg', recursive=True)
    for each in filePathList :
        print(each)
        org = load_img(each)
        x = img_to_array(org)
        x = x.reshape((1,) + x.shape)
        tmp = path.splitext(each)
        tmp2 = path.split(tmp[0])
        fileName = tmp2[1]

        train_datagen.fit(x, augment=True, seed=seedNumber)

        currentFolderPath = path.split(each)[0]
        if currentFolderPath != UsedFolderPath:
            currentFileNumber = len(os.walk(currentFolderPath).__next__()[2])
            UsedFolderPath = currentFolderPath
            #print(1)
            #print(UsedFolderPath)
            isUsedFolderPath = False
        else:
            isUsedFolderPath = True
            #print(2)
            #print(UsedFolderPath)
            # nothing
        # print(currentFileNumber)

        i = 0
        maxCount = maxBasisNumber // currentFileNumber
        print(maxCount)

        for batch in train_datagen.flow(x, batch_size=1, seed=seedNumber,
                                        save_to_dir=currentFolderPath, save_prefix=fileName, save_format='jpg'):
            i += 1
            if i > maxCount:
                break

TrainPatentPath =  'E:\\Study\\Pill\\ForegroundTest\\FirstDis\\Class0\\Train\\Class2'

MakeAugData(True, TrainPatentPath)

