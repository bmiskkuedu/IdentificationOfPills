# IdentificationOfPills
Source Code Of Identification Of Pills Using Deep Learning 

1. Purpose : Identification of pills using their pictures which are taken by Smartphone.

2. work flow

   ![image](https://user-images.githubusercontent.com/45959329/85300412-7b730c00-b4e1-11ea-93d1-157d8c103d72.png)

3. Pill Localization
   1) Delete background by semantic segmentation
      - Train/Segmentation/ModelTraining.py
   2) Seperate front and rear image of pill
      - Train/Utils/SeperateFrontBackImg.py
      
   ![image](https://user-images.githubusercontent.com/45959329/85349182-be140300-b538-11ea-90ab-bc1d8ff6c8b1.png)

4. Shape Classification
   1) Classify the shape of pills
   2) Excute binary process
   3) Training the CNN Model
