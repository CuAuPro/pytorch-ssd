from vision.datasets.voc_dataset import VOCDataset


dataset = VOCDataset("data/npdata", dataset_type="train")

dataset.get_annotation(0)
dataset.get_image(0)
dataset._read_image(dataset.ids[0])
a = 1