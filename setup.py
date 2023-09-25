from setuptools import setup, find_packages

setup(
    name="video-sync",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pytorchvideo',
        'pytorch-lightning',
        'torchvision',
    ],
)
