from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json


def register_custom_dataset(name, json_file, img_folder, thing_class=None, evaluator_type=None):
    extra_annotation_keys = ["point_coords", "point_labels"]
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, img_folder, extra_annotation_keys=extra_annotation_keys))
    metadata = {}
    if thing_class:
        metadata["thing_classes"] = [thing_class]
    if evaluator_type:
        metadata["evaluator_type"] = evaluator_type
    MetadataCatalog.get(name).set(**metadata)


def add_register_custom_dataset():
    # Register your custom datasets
    register_custom_dataset("custom_damage_train", "/content/drive/MyDrive/StatHack/PointRend/NG/train/point_coco_annotations.json", "/content/drive/MyDrive/StatHack/PointRend/NG/train/", thing_class="Damage")
    register_custom_dataset("custom_damage_val", "/content/drive/MyDrive/StatHack/PointRed/NG/val/point_coco_annotations.json", "/content/drive/MyDrive/StatHack/PointRend/NG/val/", evaluator_type="Damage")
    register_custom_dataset("custom_screw_train", "/content/drive/MyDrive/StatHack/PointRend/screw_NG/train/point_coco_annotations.json", "/content/drive/MyDrive/StatHack/PointRend/screw_NG/train/", thing_class="no_screw")
    register_custom_dataset("custom_screw_val", "/content/drive/MyDrive/StatHack/PointRed/screw_NG/val/point_coco_annotations.json", "/content/drive/MyDrive/StatHack/PointRend/screw_NG/val/", evaluator_type="no_screw")
