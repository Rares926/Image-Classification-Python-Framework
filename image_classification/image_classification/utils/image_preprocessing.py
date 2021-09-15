import cv2 as cv

#Internal framework imports 
from ..data_structures.image_shape import ImageShape #TODO: change to relative import after testing
from ..data_structures.ratio import Ratio
from ..data_structures.point import Point
#Typing imports

class ImageProcessing:


    @staticmethod
    def stretch(image, image_shape: ImageShape):

        if image is None:
            raise("Image is empty")
        old_height, old_width = image.shape[:2]

        if old_height == image_shape.height and old_width == image_shape.width:
            return image
        else:
            return cv.resize(image, (image_shape.width, image_shape.height))

    @staticmethod
    def border(image ,image_shape:ImageShape ,padding = "center"):

        (h, w) = image.shape[:2]

        if padding=="center":

                top=bottom=int((image_shape.height-h)/2)
                left=right=int((image_shape.width-w)/2)

                image = cv.copyMakeBorder(image,top, bottom, left, right, cv.BORDER_CONSTANT, None, value = 0)
                image=cv.resize(image,(image_shape.width,image_shape.height))

        elif padding=="top-left":
                bottom=image_shape.height-h
                right=image_shape.width-w
                image = cv.copyMakeBorder(image,0, bottom, 0, right, cv.BORDER_CONSTANT, None, value = 0)

        return image

    @staticmethod
    def aspect_ratio_resize(image,image_shape: ImageShape,inter = cv.INTER_AREA):

        if image is None:
            raise("Image is empty")

        (h, w) = image.shape[:2]

        aspect=min(image_shape.height/h,image_shape.width/w)

        new_dim=(int(aspect*w),int(aspect*h))

        resized = cv.resize(image, new_dim, interpolation = inter)

        return resized

    @staticmethod
    def letterbox(image, image_shape: ImageShape ,padding='center'):
        
        if image is None:
            raise("Image is empty")
        if image_shape.height is None and image_shape.width is None:
            return image
        else:
             image=ImageProcessing.aspect_ratio_resize(image,image_shape)


        bordered_image=ImageProcessing.border(image, image_shape, padding)
        return bordered_image

    @staticmethod
    def crop(image, cropped_image_shape: ImageShape, ratio:Ratio):
        if image is None:
            raise("Image is empty")
        original_height, original_width, channels = image.shape
        original_image_shape = ImageShape({'width':original_width,'height':original_height,'depth':channels})
        top_left_point, bottom_right_point = ratio.size_calculator(original_image_shape)
        cropped_image = image[top_left_point.coord_y:bottom_right_point.coord_y, top_left_point.coord_x:bottom_right_point.coord_x]
        cropped_image = ImageProcessing.stretch(cropped_image, cropped_image_shape)
        return cropped_image

    @staticmethod
    def normalize(image):
        if image is None:
            raise("Image is empty")
        normalized_image = image / 255
        return normalized_image    
        


def run():
    
    img = cv.imread("C:/Users/Radu Baciu/Desktop/workspaceTesting/inputData/train/class_000P0.jpg")
    TEST_SHAPE = ImageShape({'width':100,'height':300,'depth':3})
    TEST_SHAPE2 = ImageShape({'width':500,'height':700,'depth':3})
    TEST_SHAPE3 = ImageShape({'width':300,'height':280,'depth':3})
    TEST_RATIO = Ratio(0.3,0.3)
    cv.imshow('Original image', img)

    # img2=ImageProcessing.stretch(img, TEST_SHAPE)
    # cv.imshow("Streched image",img2)
    # cv.waitKey(0)    

    # img1=ImageProcessing.letterbox(img, TEST_SHAPE2, padding="center")
    # cv.imshow('Processed image', img1)
    # cv.waitKey(0)

    img3=ImageProcessing.crop(img,TEST_SHAPE3,TEST_RATIO)
    cv.imshow("Cropped image",img3)
    cv.waitKey(0)
    img3 = ImageProcessing.normalize(img3)
    a=0


if __name__ == "__main__":
    run()