class LabelDataset:
    def __init__(self, file):
        dataset = open(file, "rb")
        # Read header
        self.magic_num = int.from_bytes(dataset.read(4), "big")
        self.num_of_items = int.from_bytes(dataset.read(4), "big")
        # Read Images
        self.labels = []
        for _ in range(self.num_of_items):
            self.labels.append(int.from_bytes(dataset.read(1), "big"))
        
        dataset.close()

    def get_labels(self):
        return self.labels