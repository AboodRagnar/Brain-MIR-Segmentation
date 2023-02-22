#define an imbalanced dataset with a 1:100 class ratio


# from numpy import unique
# from matplotlib import pyplot as plt
# from sklearn.datasets import make_blobs

# x,y=make_blobs(1000,2,3)
# dic={}
# for i in range(len(unique(y))):
#     unqlis=unique(y)[i]
#     dic.update({unqlis: (y==unqlis)})
    
# print(x[True,0],x[True,1])
# for k,v in dic.items():
    
#     plt.scatter(x=x[v,0],y=x[v,1])
# plt.show()
from PIL import Image
import os
import json
import h5py
import numpy as np
from numpy import unique
from keras.layers import Dropout, Dense,Flatten,Conv2D,MaxPooling2D,GlobalAveragePooling2D,Input
from keras.optimizers import Adam
from keras.models import Model,load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plot
from keras.applications import VGG19
import cv2
from keras.callbacks  import EarlyStopping
pathTrain='D:/AI/Keras/DataSet/Brain MIR Segmentation/lgg-mri-segmentation/kaggle_3m/Train'
pathTest='D:/AI/Keras/DataSet/Brain MIR Segmentation/lgg-mri-segmentation/kaggle_3m/Test'
pathPredict='D:/AI/Keras/DataSet/Brain MIR Segmentation/lgg-mri-segmentation/kaggle_3m/Predict'

class  ModelingData:
    
    def  __init__(self):
      self.Train = []
      self.Test = []
      self.Predict=[]
      self.NNOutput=2
      self.TrainModel=[]
      self.LoadingModel=[]
    
    def fix_layer0(self,filename, batch_input_shape, dtype):
        with h5py.File(filename, 'r+') as f:
            model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
            layer0 = model_config['config']['layers'][0]['config']
            layer0['batch_input_shape'] = batch_input_shape
            layer0['dtype'] = dtype
            f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
    def SavImgToDiffierentDirectoy(self,oldpath, newpath,imageformat='Png',convertimgformat='RGB'):

       
        
        for img in os.listdir(oldpath):
            if img.endswith('.'+imageformat): 
                place_holder_path=oldpath+'/'+img
                print(place_holder_path)
                GetImg=Image.open(place_holder_path).convert(convertimgformat)
                GetImg.save(newpath+'/'+img)
    
    def DeleteDuplicatNamedImagesDirectories(self,imgpathdeleting=pathTrain+'/Tumor',imgpathcomparing=pathTest+'/Tumor'):
        for img in os.listdir(imgpathdeleting):
            if img in os.listdir(imgpathcomparing):
                print(img)
                os.remove(imgpathdeleting+'/'+img)

    def ParperImagesForTraining(self):
        GetImag=ImageDataGenerator()
        GetTrainImg=GetImag.flow_from_directory(pathTrain,target_size=(224,224),classes=['Tumor','NoTumor'],batch_size=32)
        GetTestImg=GetImag.flow_from_directory(pathTest,target_size=(224,224),classes=['Tumor','NoTumor'],batch_size=32)
        GetPredictImg=GetImag.flow_from_directory(pathPredict,target_size=(224,224),classes=['Tumor','NoTumor'],batch_size=15)
        self.Train = GetTrainImg
        self.Test = GetTestImg
        self.Predict=GetPredictImg
        

    def CreatVGG19Model(self,outputneuron, optimizer=Adam(),loss='categorical_crossentropy'):
        
       
        #Vgg19=VGG19()
        # Model=Sequential()
        # for ly in Vgg19.layers[:-1]:
        #     ly.trainable=False
        #     Model.add(ly)
        # Model.add(Dropout(0.4))
        
        #Model.add(Dense(outputneuron,activation='softmax'))
        #Model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        
        #return Model  
        Vgg19=VGG19(input_shape=[224,224,3],include_top=False, weights='imagenet')
        x=Flatten()(Vgg19.output)
        x=Dense(2,activation='sigmoid')(x)
        
        model=Model(Vgg19.input,x)
        model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        
        return model  


    
    def TrainModelUsingGenerator(self,stepperepoch=50,epochs=15):
        model=self.CreatVGG19Model(self.NNOutput)
        self.ParperImagesForTraining()

        model.fit_generator(self.Train,verbose=1,validation_data=self.Test,shuffle=True,callbacks=[EarlyStopping(monitor='val_loss',patience=3)])
        model.save('LGGTumorOrNot.h5')
        # plot.plot(self.TrainModel.history.history['accuracy'],'r') # verbose must be False to work
        # plot.title('Loss =')
        # plot.show()
        self.TrainModel=model

   

        
    def LoadModel(self, modelname):
        self.fix_layer0(modelname,[None, 224, 224, 3], 'float32')
        self.LoadingModel=load_model(modelname)
        
    def PlotImageBatch(self,image_batch,figure_size=(12,16), loadmodel='None'):
        image_array,labels=next(image_batch)

        fig=plot.figure(figsize=figure_size)

        if loadmodel!='None':
            self.LoadModel(loadmodel)
            scor=self.LoadingModel.evaluate_generator(image_batch,steps=5)
            print(image_batch.class_indices)
            #print(self.LoadingModel.predict_classes(image_array))

            print(self.LoadingModel.predict(image_array))
            print(scor[0],scor[1])
       
        for i in range(len(image_array)):
            fig.add_subplot(5,3,i+1).axis('Off')
            image = np.array(image_array[i]).astype(np.uint8)
            
            plot.title(str(i)+str(labels[i]))
            plot.imshow(image)
         
        
        plot.show()
  




DataModel=ModelingData()
DataModel.TrainModelUsingGenerator()

DataModel.PlotImageBatch(DataModel.Predict,loadmodel='LGGTumorOrNot.h5')



