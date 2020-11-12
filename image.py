import numpy as np

class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = []
        
    
    def getSize(self):
        return self.width * self.height
    
    
    def addPixel(self, pixel):
        self.pixels.append(pixel)
        
        
    def getPixelsArray(self):
        return np.reshape(np.array(self.pixels), (self.width, self.height))
        
        
        # pixels_2D = []
        # current_pixel = 0
        
        # for _ in range(self.width):
        #     pixel_row = []
        #     for _ in range(self.height):
        #         pixel_row.append(self.pixels[current_pixel])
        #         current_pixel += 1
            
        #     pixels_2D.append(pixel_row)
        
        # return np.array(pixels_2D)
    
    
    # may not be needed
    '''
    def setPixel(self, index, pixel):
        if abs(index) >= self.getSize():
            return False
        
        self.pixels[index] = pixel
        return True

    
    def getPixel(self, index):
        if abs(index) >= self.getSize():
            return -1
    
        return self.pixels[index]
    '''
    
