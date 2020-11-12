from image import Image
import numpy as np

class Dataset:
    def __init__(self, file):
        dataset = open(file, "rb")
        # Read header
        self.magic_num = int.from_bytes(dataset.read(4), "big")
        self.num_of_images = int.from_bytes(dataset.read(4), "big")
        self.num_of_rows = int.from_bytes(dataset.read(4), "big")
        self.num_of_columns = int.from_bytes(dataset.read(4), "big")

        # Read Images
        self.images = []
        for _ in range(self.num_of_images):
            # Create image object
            img = Image(self.num_of_columns, self.num_of_rows)
            # Read pixels of current image
            for _ in range(self.num_of_rows * self.num_of_columns):
                img.addPixel(int.from_bytes(dataset.read(1), "big"))

            self.images.append(img)
        
        dataset.close()


    def getImageDimensions(self):
        return (self.num_of_rows, self.num_of_columns)


    def getImages(self):
        images = []
        for img in self.images:
            images.append(img.getPixelsArray())
            
        return np.array(images)