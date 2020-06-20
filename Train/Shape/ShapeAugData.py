from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os.path as path
import os
import glob

data_gen_args = dict(rotation_range=5.,
                     #zoom_range=0.05,
                     #horizontal_flip = True,
                     #vertical_flip = True
                     #brightness_range=[0.9,1.1],
                     )

def GetAugNumber(folderPath):
    FilePathList = glob.glob(path.join(folderPath, '*.*'))
    augNumber = 2000 // len(FilePathList)

    return augNumber

def Excute():
    train_datagen = ImageDataGenerator(**data_gen_args)
    TargetPath =  "E:\\Study\\Pill\\Shape\\data\\"
    # FilePathList = glob.glob(path.join(TargetPath, '*.*'))
    FilePathList = glob.iglob(TargetPath + '/**/*.jpg', recursive=True)
    prevFolderPath = ''
    augNumber = 0
    seedNumber = 5
    for each in FilePathList:
        print(each)
        org = load_img(each)
        x = img_to_array(org)
        x = x.reshape((1,) + x.shape)
        train_datagen.fit(x, augment=True, seed=seedNumber)

        tmp = path.splitext(each)
        tmp2 = path.split(tmp[0])
        fileName = tmp2[1]

        folerPath = path.split(each)[0]
        if prevFolderPath != folerPath:
            #augNumber = GetAugNumber(folerPath)
            augNumber = 150
            print(augNumber)
            prevFolderPath = folerPath

        i = 0
        maxCount = augNumber

        for batch in train_datagen.flow(x, batch_size=1, seed=seedNumber,
                                        save_to_dir=folerPath, save_prefix=fileName, save_format='jpg'):
            i += 1
            if i > maxCount:
                break




if __name__ == "__main__" :

    import os.path as path
    import glob
    Excute()