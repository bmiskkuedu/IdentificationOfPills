# 결과 이미지가 입력의 반으로 줄어듦(이유는 모름)
# 따라서 모델에다가 입력 크기를 원본보다 두배로 설정하면 결과가 원본 이미지와 같은 크기가 나옴.
# 크기는 32의 배수가 되어야 함. assert input_height%32 == 0
# 학습시 색 정보가 중요 한 것 같음.
# 배경과 전경의 색이 비슷하면 결과가 좋지 않음.

#from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.unet import resnet50_unet


#model = vgg_unet(n_classes=2 ,  input_height=640, input_width=640)
model = resnet50_unet(n_classes=2 ,  input_height=640, input_width=640)

model.train(
    train_images =  "E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\Training\\Src\\",
    train_annotations = "E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\Training\\Mask\\",
    checkpoints_path = "E:\\Study\\Pill\\SegTest\\Data\\firstDis2\\Model\\Resnet50" , epochs=5

)


