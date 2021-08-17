import cv2 as cv
import numpy as np
# Internal framework imports

# Typing imports imports

class ImageProcessing:

    def __init__(self):
        pass

    @staticmethod
    def strech(image,height,width=0):

        if image is None:
            raise("Image is empty")

        old_height, old_width = image.shape[:2]
        
        if old_height==height and old_width==width:
            return image
        elif width==0:
            return cv.resize(image, (height, height))
        else:
            return cv.resize(image, (height, width))

    @staticmethod
    def letterbox(image,height,width,inter = cv.INTER_AREA):

        (h, w) = image.shape[:2]

        aspect=min(height/h,width/w)

        new_dim=(int(aspect*w),int(aspect*h))

        resized = cv.resize(image, new_dim, interpolation = inter)

        return resized

    @staticmethod
    def aspect_ratio_resize(image, width = None, height = None, inter = cv.INTER_AREA):

        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized







def run():
    # C:\Users\test\Desktop
    img = cv.imread("C:/Users/test/Desktop/550&320.png")
    cv.imshow('Original image', img)
    cv.waitKey(0)
    # height, width = img.shape[:2] # shape ul trebuie salvbat in alta clasa in functie de tipul imaginii rgb ,greyscale etc 
    # img=ImageProcessing.strech(img,800,200)
    # img = ImageProcessing.aspect_ratio_resize(img,700)
    img=ImageProcessing.letterbox(img,300,500)
    # print(img)
    cv.imshow('Processed image', img)
    cv.waitKey(0)






if __name__ == "__main__":
    run()