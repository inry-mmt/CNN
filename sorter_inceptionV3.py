import os
import re
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
from random import randint, random
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def main():
    """
    text
    """

    print("")

class Sorter():
    img_rows = 300
    img_cols = 300
    # classes = ['empty', 'notempty']
    channels = 3
    batch_size = 50

    n_epoch = 100

    def __init__(self, classes=[], train_dir="", validation_dir="", save_weights_path="./inceptionv3_finetuning.h5", finetuning_weights_path="", img_size=(300, 300)):
        if not hasattr(classes, "__iter__"):
            raise ValueError("classes には iterable なオブジェクトを与えてください")
        elif len(classes) < 1:
            raise ValueError("classes が空です")

        self.__preloaded_model = None
        self.classes = classes
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.save_weights_path = save_weights_path
        self.finetuning_weights_path = finetuning_weights_path
        self.img_rows = img_size[0]
        self.img_cols = img_size[1]

    def train(self):
        if self.train_dir == "" or self.validation_dir == "":
            raise ValueError("train_dir と validation_dir に画像が入ったディレクトリへのパスを入力してください")

        img_rows = self.img_rows
        img_cols = self.img_cols
        classes = self.classes
        n_classes = len(classes)
        channels = self.channels
        batch_size = self.batch_size
        n_epoch = self.n_epoch

        n_train_samples = self._count_files(self.train_dir)
        n_val_samples = self._count_files(self.validation_dir)

        print(n_train_samples)
        print(n_val_samples)

        trainImageGenerator = ImageDataGenerator(
            rescale = 1 / 255,
#            samplewise_std_normalization=True,
#            samplewise_center=True,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=self._preprocess,
#            width_shift_range=0.05,
#            height_shift_range=0.05,
        )

        train_generator = trainImageGenerator.flow_from_directory(
            directory=self.train_dir,
            target_size=(img_rows, img_cols),
            color_mode='rgb',
            classes=classes,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
#            save_to_dir="./generated_pic",
        )

        validationImageGenerator = ImageDataGenerator(
            rescale = 1 / 255,
#            preprocessing_function=self._preprocess,
#            samplewise_std_normalization=True,
#            samplewise_center=True,
        )

        validation_generator = validationImageGenerator.flow_from_directory(
            directory=self.validation_dir,
            target_size=(img_rows, img_cols),
            color_mode='rgb',
            classes=classes,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True
        )

        if self.finetuning_weights_path:
            model = self.model(weights_path=self.finetuning_weights_path)
        else:
            model = self.model()

        # define callbacks
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=0,
            mode="auto",
        )

        # define TensorBoard
        tensorboard_callback = TensorBoard(
            log_dir="./logs",
        )

        # tuning
        history = model.fit_generator(
            train_generator,
            samples_per_epoch=n_train_samples,
            nb_val_samples=n_val_samples,
            nb_epoch=n_epoch,
            validation_data=validation_generator,
            callbacks=[tensorboard_callback, early_stopping_callback],
        )

        model.save_weights(self.save_weights_path)

    def evaluate(self):
        if self.validation_dir == "":
            raise ValueError("validation_dir に画像が入ったディレクトリへのパスを入力してください")

        img_rows = self.img_rows
        img_cols = self.img_cols
        classes = self.classes
        n_classes = len(classes)
        channels = self.channels
        batch_size = self.batch_size
        n_epoch = self.n_epoch

        n_val_samples = self._count_files(self.validation_dir)

        print(n_val_samples)


        validationImageGenerator = ImageDataGenerator(
            rescale = 1 / 255,
#            samplewise_std_normalization=True,
#            samplewise_center=True,
        )

        validation_generator = validationImageGenerator.flow_from_directory(
            directory=self.validation_dir,
            target_size=(img_rows, img_cols),
            color_mode='rgb',
            classes=classes,
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True
        )

        if self.finetuning_weights_path:
            model = self.model(weights_path=self.finetuning_weights_path)
        else:
            model = self.model()

        # tuning
        history = model.evaluate_generator(
            validation_generator,
            steps=n_val_samples
        )

        print("accuracy: {}".format(history[1]))

    def preload_model(self):
        self.__preloaded_model = self.model(self.finetuning_weights_path)

        return self.__preloaded_model

    def detect(self, filename=""):
        if self.__preloaded_model is None:
            model = self.preload_model()
        else:
            model = self.__preloaded_model

        classes = self.classes

        img = load_img(filename, target_size=(300, 300))
        x = img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        x = x / 255

        x = np.asarray([x])
        pred = model.predict(x)[0]

        top = pred.argsort()[::-1][0]

        return classes[top]


    def model(self, weights_path=None, n_freeze=20):
        img_rows = self.img_rows
        img_cols = self.img_cols
        classes = self.classes
        n_classes = len(classes)

        # model
        #input_tensor = Input(shape=(img_rows, img_cols, 3))
        #inception_v3 = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)
        inception_v3 = InceptionV3(include_top=False, weights="imagenet", input_shape=(img_rows, img_cols, 3))

        top_model = Sequential()
        top_model.add(Flatten(input_shape=inception_v3.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(n_classes, activation='softmax'))

        # attach
        model = Model(inputs=[inception_v3.input], outputs=[top_model(inception_v3.output)])

        # load weights
        if weights_path is not None:
            model.load_weights(weights_path)

        # freeze weights
        # https://github.com/danielvarga/keras-finetuning/blob/master/train.py
        # how_many = 172
        how_many = n_freeze
        for layer in model.layers[:how_many]:
            layer.trainable = False
        for layer in model.layers[how_many:]:
            layer.trainable = True

        # compile
        lr = 1e-4 / 2
        momentum = 0.9
        model.compile(loss='categorical_crossentropy',
                                  optimizer=optimizers.SGD(lr=lr, momentum=momentum),
                                    metrics=['accuracy'])


        return model

    def _count_files(self, base_dir):
        def _is_imagefile(filename):
            if re.match(r'.+\.(png|jpg|jpeg|bmp|tiff)$', filename):
                return True
            else:
                return False

        count = 0
        for category in self.classes:
            files = os.listdir(os.path.join(base_dir, category))
            files = [f for f in files if _is_imagefile(f)]
            count += len(files)

        return count

    def _preprocess(self, tensor):
        # ランダムに色味を変化
        #h_range = 0.05
        h_range = 0.01
        #s_range = 0.4
        s_range = 0.1
        #v_range = 60
        v_range = 20

        # HSVに変化 [0~1, 0~1, 0~255], dtype=float32
        tensor_hsv = rgb_to_hsv(tensor)
        tt = tensor_hsv.T
        h = tt[0]
        s = tt[1]
        v = tt[2]

        # 色相を変化 -h_range - s_range の間
        # clipして 0 - 1 の間からはみださないように
        h = np.clip(h + (random() * (h_range * 2) - h_range), 0, 1)

        # 彩度を変化 -s_range ~ s_range の間
        # 同上
        s = np.clip(s + (random() * (s_range * 2) - s_range), 0, 1)

        # 明度を変化
        v = np.clip(v + randint(-v_range, v_range), 0, 255)

        rs = self.img_rows
        cs = self.img_cols
        tensor_hsv = np.concatenate(
            (
                h.reshape(1, rs, cs),
                s.reshape(1, rs, cs),
                v.reshape(1, rs, cs),
            ),
            axis=0).T

        # 結果出力
        result = hsv_to_rgb(tensor_hsv)
        return result



if __name__ == '__main__':
    main()
