import cv2   as cv
from ..data_structures.image_loader import ImageLoader
from ..data_structures.resize_method import ResizeMethod
from ..network.data_generator import DataGenerator
from ..utils.helpers.json_helper import JsonHelper
import numpy as np
import os
import tensorflow as tf

# Internal framework imports
from ..utils.data_processing       import DataProcessing
from ..utils.helpers.io_helper     import IOHelper
from ..builders.network_builder    import NetworkBuilder

# Typing imports imports


class TestWorker:
    def __init__(self, model, data_file_location):
        self.model = model
        self.data_file_location = data_file_location

    def load_checkpoint(self,checkpoint_path:str):
        if self.model is None:
            raise Exception("The model must be created in order to be used!")

        self.model.summary()
        IOHelper.check_if_file_exists(checkpoint_path, "Checkpoint path invalid: ")
        self.model.load_weights(checkpoint_path)


    def top_k_batch(self,predictions,truths,topK):

        tmp = np.zeros_like(predictions)
        tmp[np.arange(len(predictions)), predictions.argmax(1)] = 1
        result=0
        for idx in range(len(truths)):
            tb_verrified=truths[idx]
            if tb_verrified in tmp[idx][:topK]:
                result+=1
        
        return result
            


    def test_images(self, images_path:str, network: NetworkBuilder, results_folder, preprocess_images: bool,topK: int): 
        image_loader = ImageLoader(network.image_shape, network.image_format, network.resize_method, network.ratios, network.resize_after_crop)
        if preprocess_images == False:
            image_loader.resize_method = ResizeMethod.NONE
            
        label_list = JsonHelper.read_json(self.data_file_location)
        label_names = DataProcessing.load_label_names(self.data_file_location)

        testing_generator = DataGenerator(images_path, label_list, image_loader, network.batch_size, is_train_data = False)

        topK_predicted=0
        topK_total=0

        for index in range(len(testing_generator)):

            batch_images, ground_truths, image_names = testing_generator[index]

            raw_predictions = self.model.predict(batch_images, testing_generator.batch_size) #aici trebuie sa iau pe rand fiecare rand si la maxim sa pun 1 iar la minim 0
            predictions = [np.argmax(i) for i in raw_predictions]

            ground_truths = ground_truths.astype(int)

            topK_predicted+=self.top_k_batch(raw_predictions,ground_truths,topK)
            topK_total+=len(batch_images)
            for index in range(len(batch_images)):
                if predictions[index] != ground_truths[index]:
                    image_data = batch_images[index] * 255
                    path = os.path.join(results_folder, label_names[predictions[index]], image_names[index])
                    cv.imwrite(path, image_data)

        print('Done')
        print('TopK accuracy for topk={} is {}'.format(topK,topK_predicted/topK_total))
            

      

