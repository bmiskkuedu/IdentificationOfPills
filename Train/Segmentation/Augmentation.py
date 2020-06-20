# To Augment the Train Data
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os.path as path
import glob

# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip = True,
                     vertical_flip = True
                     #brightness_range=[0.9,1.1],
                     )
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 5

SourcePath = "E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\Training\\Src\\"
TargetPath = SourcePath
FilePathList = glob.glob(path.join(TargetPath, '*.jpg'))
MaskParentPath = "E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\Training\\Mask\\"
for each in FilePathList:
    print(each)
    org = load_img(each)
    x = img_to_array(org)
    x = x.reshape((1,) + x.shape)
    image_datagen.fit(x, augment=True, seed=seed)

    # each에서 확장자를 제외한 파일 이름을 가져온다.
    tmp = path.splitext(each)
    tmp2 = path.split(tmp[0])
    fileName = tmp2[1]

    maskPath = MaskParentPath + fileName +".png"

    mask = load_img(maskPath)
    y = img_to_array(mask)
    y = y.reshape((1,) + y.shape)
    mask_datagen.fit(y, augment=True, seed=seed)

    i = 0
    maxCount = 500
    for batch in image_datagen.flow(x, batch_size=1, seed=seed,
                              save_to_dir=SourcePath, save_prefix=fileName, save_format='jpeg'):
        i += 1
        if i > maxCount:
            break

    i = 0
    for batch in mask_datagen.flow(y, batch_size=1, seed=seed,
                              save_to_dir=MaskParentPath, save_prefix=fileName, save_format='png'):
        i += 1
        if i > maxCount:
            break



