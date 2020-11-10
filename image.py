class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = []
        
    
    def getSize(self):
        return self.width * self.height
    
    
    def addPixel(self, pixel):
        self.pixels.append(pixel)
    
    # may not be needed
    def setPixel(self, index, pixel):
        if abs(index) >= self.getSize():
            return False
        
        self.pixels[index] = pixel
        return True

    
    def getPixel(self, index):
        if abs(index) >= self.getSize():
            return -1
    
        return self.pixels[index]