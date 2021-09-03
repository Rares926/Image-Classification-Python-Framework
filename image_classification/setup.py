from setuptools import setup,find_packages


setup(
    name="image_classification",
    version="0.0.1",
    author="",
    description="Classification",
    url="",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires = [
            'opencv-python==4.5.3.56',
            'tensorflow==2.5.0',
            'seaborn==0.11.1',
            'scikit-learn==0.24.2',
            'jsonargparse==3.4.1',
            'albumentations==1.0.3',
            'tensorflow-hub==0.12.0'
      ],
      entry_points={
            'console_scripts': [
                  'image_classif_train= image_classification.train:run',
                  'image_classif_test= image_classification.test:run'
            ],
      }
)