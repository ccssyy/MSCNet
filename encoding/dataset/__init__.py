from .base import *
from .cityscapes import CityscapesSegmentation
from .ISIC_data import ISICImageDataset, ISIC2017Dataset, ISIC2016Dataset, PH2Dataset

datasets = {
    'cityscapes': CityscapesSegmentation,
    'isic2018': ISICImageDataset,
    'isic2017': ISIC2017Dataset,
    'isic2016': ISIC2016Dataset,
    'ph2': PH2Dataset,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
