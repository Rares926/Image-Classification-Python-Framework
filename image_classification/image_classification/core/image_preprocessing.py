import cv2 as cv
import os
# Internal framework imports

# Typing imports imports

class ImagePreprocessing:

    def __init__(self):
        pass

    @staticmethod
    def strech(image,size):
        return cv.resize(image, (size, size))

    @staticmethod
    def crop():
        print("c")

    @staticmethod
    def letterbox():
        print("l")


def run():
    img_arr = cv.imread("C:/Users/test/Desktop")
    cv.imshow("img_arr")
    # cv.imshow()






if __name__ == "__main__":
    run()