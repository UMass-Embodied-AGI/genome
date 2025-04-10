from engine.registry import registry

from engine.datasets.base_dataset import BaseDataset
from engine.datasets.gqa_dataset import GQADataset
from engine.datasets.refcoco_dataset import RefcocoDataset
from engine.datasets.imgedit_dataset import IMGEDITDataset
from engine.datasets.okdet_dataset import OKDETDataset


def get_dataset(name):
    return registry.get_dataset_class(name)
