
from image_classification.builders.network_builder import NetworkBuilder
import tensorflow     as tf
import datetime
import os
import albumentations as A

# Internal framework imports
from ..utils.helpers.io_helper      import IOHelper
from ..network.data_generator       import DataGenerator
from ..data_structures.image_loader import ImageLoader
from ..network.confusion_matrix     import ConfusionMatrixCallback

# Typing imports imports

class TrainWorker:
    
    def __init__(self, model, network: NetworkBuilder, starting_epoch:int=0):
        self.model=model
        self.starting_epoch=starting_epoch
        self.network=network


    def train(self, workspace:str, labels, image_loader:ImageLoader, from_checkpoint:str=None):
        if self.model is None:
            raise Exception("The model must be created in order to be used!")

        train_location = os.path.join(workspace, 'inputData', 'train')
        test_location = os.path.join(workspace, 'inputData', 'test')

        labels_location = os.path.join(workspace, 'data.json')
        checkpoint_path = os.path.join(workspace,'checkpoints')
        
        IOHelper.create_directory(checkpoint_path)

        print("-------------------->CREATING CALLBACKS<--------------------")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"cp-{epoch:03G}.h5",
                                                         save_best_only=False,
                                                         save_freq='epoch',
                                                         monitor='val_loss',
                                                         save_weights_only=True,
                                                         verbose=1)

        workspace=workspace+"/tensorboard/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=workspace, histogram_freq=1)
        print("-------------->CALLBACKS CREATED SUCCESSFULLY<--------------")


        print("--------------------->COMPILING MODEL<----------------------")
        self.model.compile( optimizer=self.network.optimizer,
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
                            metrics=self.network.metrics )
        print("--------------------->MODEL COMPILED<-----------------------")


        if from_checkpoint!=None:
            self.model.load_weights(from_checkpoint)
            print("--------------->WEIGHTS LOADED FROM CHECKPOINT<---------------")
                    

        transform = A.Compose(self.network.augmentations)

        print("------------------->SHOW MODEL SUMMARY<---------------------")
        self.model.summary()


        print("--------->CREATING TRAINING AND TESTING GENERATORS<---------")
        training_generator = DataGenerator(train_location, labels, image_loader, self.network.batch_size, is_train_data=True, transform = transform)
        testing_generator = DataGenerator(test_location, labels, image_loader, self.network.batch_size, is_train_data=True)
        print("-------------->PROCCES COMPLETED SUCCESSFULLY<--------------")


        print("-------------------->STARTING TRAINING<---------------------")
        self.model.fit(training_generator,validation_data = testing_generator, epochs=self.network.epochs,initial_epoch=self.starting_epoch, callbacks=[tensorboard_callback,ConfusionMatrixCallback(self.model, testing_generator, workspace, labels_location),cp_callback])
        print("-------------->TRAINING COMPLETED SUCCESSFULLY<--------------")


        print("------------------->PRINTING TEST RESULTS<-------------------")
        test_results = list(self.model.evaluate(testing_generator, verbose=1))
        
        for item in test_results:
            print(item)