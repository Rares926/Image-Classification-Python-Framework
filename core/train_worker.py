import tensorflow as tf

# Internal framework imports

# Typing imports imports


class TrainWorker:
    def __init__(self, model_layers, augment_layers = None):
        self.model = model_layers
        self.augments = augment_layers

    def create_model(self, labels):
        # self.model = tf.keras.models.Sequential([
        #     self.augments,
        #     self.model,
        #     tf.keras.layers.Dense(len(labels), activation='softmax')
        # ])

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(labels), activation='softmax')
        ])

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.model.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy'])

    def train(self, workspace, x_train, y_train, x_test, y_test, epochs = 10):
<<<<<<< Updated upstream
        if self.model is None:
            raise Exception("The model must be created in order to be used!")

        self.model.fit(x_train, y_train, epochs=epochs)
=======

        checkpoint_path = os.path.join(workspace,"checkpoints")
        IOHelper.create_directory(checkpoint_path)

        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"{epoch:03G}cp.h5",
                                                         save_best_only=False,
                                                         save_freq='epoch',
                                                         monitor='val_loss',
                                                         save_weights_only=True,
                                                         verbose=1)


        workspace=workspace+"/tensorboard/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=workspace, histogram_freq=1)


        self.model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard_callback, cp_callback])
>>>>>>> Stashed changes

        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=1)
        print('\nTest loss:', test_loss)
        print('\nTest accuracy:', test_acc)