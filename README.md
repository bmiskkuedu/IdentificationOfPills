# Identification Of Pills
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
      - Example of classification in pill's shapes
      
   ![image](https://user-images.githubusercontent.com/45959329/85349950-c40ae380-b53a-11ea-9ff3-05ff026c6618.png)
   
   2) Excute binary process
      - Train/Shape/ShapePreprocess.py
      
   ![image](https://user-images.githubusercontent.com/45959329/85349937-bce3d580-b53a-11ea-9784-be5173f7f002.png)
   
   3) Training the CNN Model
      - Train/Shape/ModelTraining.py
      
5. Color Classification
   1) Classify the color of pills
      - Train/Utils/ETCUtils.py/CopySameColors
      - Example of classification in pill's color
      
   ![image](https://user-images.githubusercontent.com/45959329/85350335-dafe0580-b53b-11ea-9c74-c0f185287f4c.png)
   
6. Imprint Classification
   1) Process of Imprint Classification
   
   ![image](https://user-images.githubusercontent.com/45959329/85351723-5e6d2600-b53f-11ea-8f67-96287e79e772.png)
   
   2) Preprocess of Imprint
      - Train/Imprint/Utils/ExcuteCroping
   
   ![image](https://user-images.githubusercontent.com/45959329/85351034-b014b100-b53d-11ea-9f46-97110c07ff88.png)
   
   3) Model Training
      - Train/Imprint/ModelTraining.py 
      - We use the exist model InceptionResnetV2.

7. Test
   1) Make Lookup Table
      - Example of Lookup table structure
      
      ![image](https://user-images.githubusercontent.com/45959329/85364493-3345ff00-b55e-11ea-80ea-04a256dd6c4a.png)
      
   2) Input the user image
      - Test/ExcutePillRecognization.py
      
8. Environment

   The experiments and data analysis were carried out using Python 3.7.3 with the following openly available libraries: tensorflow 1.13.1, keras 2.2.4, 
   sklearn 0.21.2, opencv 4.1.2, and numpy 1.17.0. The pre-trained weights models were based on the Keras neural network library available at https://keras.io/api/applications/. 
