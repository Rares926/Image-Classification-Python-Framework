import cv2 as cv
import matplotlib.pyplot as plt 
from PIL import Image

class ImageProcessing:

    @staticmethod
    def strech(image, dimension = None, height = None, width = None):

        if image is None:
            raise("Image is empty")
        elif dimension is None:

            old_height, old_width = image.shape[:2]

            if old_height==height and old_width==width:
                return image
            else:
                return cv.resize(image, (width, height))
        else:
            return cv.resize(image, (dimension, dimension))

    @staticmethod
    def border(image ,height ,width ,padding = "center"):

        (h, w) = image.shape[:2]

        if padding=="center":

                top=bottom=int((height-h)/2)
                left=right=int((width-w)/2)

                image = cv.copyMakeBorder(image,top, bottom, left, right, cv.BORDER_CONSTANT, None, value = 0)
                image=cv.resize(image,(width,height))

        elif padding=="top-left":
                bottom=height-h
                right=width-w
                image = cv.copyMakeBorder(image,0, bottom, 0, right, cv.BORDER_CONSTANT, None, value = 0)

        return image

    @staticmethod
    def aspect_ratio_resize(image,height=None,width=None,inter = cv.INTER_AREA):

        (h, w) = image.shape[:2]

        aspect=min(height/h,width/w)

        new_dim=(int(aspect*w),int(aspect*h))

        resized = cv.resize(image, new_dim, interpolation = inter)

        return resized

    @staticmethod
    def letterbox(image,dimension=None,height=None,width=None,padding='center'):
        
        if image is None:
            raise("Image is empty")
        elif dimension != None:
            image=ImageProcessing.aspect_ratio_resize(image,dimension,dimension)
        elif height is None and width is None:
            return image
        else:
             image=ImageProcessing.aspect_ratio_resize(image,height,width)


        bordered_image=ImageProcessing.border(image,height,width,padding)
        return bordered_image

    @staticmethod
    def crop(image, x_axis = None, y_axis = None, height = None, width = None, n_height = None, n_width = None):

        crop = image[y_axis:y_axis+height, x_axis:x_axis+width]
        
        crop=ImageProcessing.letterbox(crop,height=n_height,width=n_width,padding="top-left")

        return crop
        


def run():
    img = cv.imread("C:/Users/test/Desktop/550&320.png")

    cv.imshow('Original image', img)
    cv.waitKey(0)

    img1=ImageProcessing.letterbox(img,height=500,width=700,padding="center")
    cv.imshow('Processed image', img1)
    cv.waitKey(0)

    img2=ImageProcessing.strech(img,height=300,width=400)
    cv.imshow("Streched image",img2)
    cv.waitKey(0)

    img3=ImageProcessing.crop(img,100,100,300,300,320,550)
    cv.imshow("Cropped image",img3)
    cv.waitKey(0)


if __name__ == "__main__":
    run()