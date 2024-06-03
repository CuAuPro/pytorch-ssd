from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.std = std

        self.augment = self._get_augmentation()

    def _get_augmentation(self):
        return Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: bounding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self._augment(img, boxes, labels)

    def _augment(self, img, boxes, labels):
        img, boxes, labels = self.augment(img, boxes, labels)
        img /= self.std
        return img, boxes, labels


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.size = size
        self.mean = mean
        self.std = std

        self.transform = self._get_transform()

    def _get_transform(self):
        return Compose([
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self._transform(image, boxes, labels)

    def _transform(self, image, boxes, labels):
        image, boxes, labels = self.transform(image, boxes, labels)
        image /= self.std
        return image, boxes, labels


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.size = size
        self.mean = mean
        self.std = std

        self.transform = self._get_transform()

    def _get_transform(self):
        return Compose([
            Resize(self.size),
            SubtractMeans(self.mean),
            ToTensor(),
        ])

    def __call__(self, image):
        return self._transform(image)

    def _transform(self, image):
        image, _, _ = self.transform(image)
        image /= self.std
        return image
