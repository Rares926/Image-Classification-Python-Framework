import cv2 as cv
import numpy as np
import os


# Internal framework imports
from .data_visualization import DataVisualization
from ..utils.image_shape import ImageShape
from ..utils.io_helper import IOHelper
# Typing imports imports


class TestWorker:
    NETWORK_SIZE = 224

    def __init__(self,model):
        self.model = model
    
    def procces_image(self,image_path:str, image_shape: ImageShape):
        image=cv.imread(image_path)[...,::-1]
        image = cv.resize(image, (image_shape.width, image_shape.height))
        image=image/255.0
        return image

    def procces_folder(self,img_names:str,folder_path:str, image_shape: ImageShape):
        data=[]
        for name in img_names:
            tmp_path=os.path.join(folder_path,name)
            image=self.procces_image(tmp_path,image_shape)
            data.append(image)

        return np.array(data)

    def load_checkpoint(self,checkpoint_path:str):
        if self.model is None:
            raise Exception("The model must be created in order to be used!")

        self.model.summary()
        IOHelper.check_if_file_exists(checkpoint_path, "Checkpoint path invalid: ")
        self.model.load_weights(checkpoint_path)

    def evaluate_model(self,test_images,test_labels):
        loss, acc = self.model.evaluate(test_images, test_labels, verbose=2)
        print("Loaded model, accuracy: {:5.2f}%".format(100 * acc))

    def test_image(self,image_path:str,image_shape: ImageShape): 
        img_names = os.listdir(image_path)
        data=self.procces_folder(img_names,image_path,image_shape)

        if not img_names:
            raise Exception("The folder is empty")
        else:
            if len(data)==1:

                result=self.model.predict(data,batch_size=1)
                max_percent_index=np.argmax(result[0])

                DataVisualization.showImage((data*255.0)[0],max_percent_index,result[0][max_percent_index])
                print("A fost detectata clasa {} cu probabilitatea {}".format(max_percent_index,result[0][max_percent_index]))

            else:
                result=self.model.predict(data,batch_size=len(img_names))
                print(result)
                indexes_list=[np.argmax(i) for i in result]
                percent=[result[index,indexes_list[index]] for index in range(len(indexes_list))]
                print(list(zip(img_names,indexes_list,percent)))
                # for index in range(len(indexes_list)):
                #     print("Pentru poza {} a fost detectata clasa {} cu probabilitatea {}".format(img_names[index],indexes_list[index],result[index,indexes_list[index]]))   