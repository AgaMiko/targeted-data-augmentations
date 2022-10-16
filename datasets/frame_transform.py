import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import glob
import numpy

class RandomFrameTransform:
    """Rotate by one of the given angles."""

    def __init__(self, mask_dir,
                image_type = "jpg",
                p=0.5,
                rotate=True,
                mask_nr='random'):
        self.mask_list =  glob.glob(mask_dir + "*."+ image_type)
        self.len = len(self.mask_list)
        self.p = float(p)
        self.rotate = rotate
        self.mask_nr = mask_nr
        
    def __call__(self, image, **params):
        p = random.randint(0, 100)
        if p <= self.p*100:
            if self.rotate:
                angle = random.randint(0, 360)
            if self.mask_nr in ['random', None]:
                mask_nr =  random.randint(0, self.len-1)
            else:
                mask_nr = self.mask_nr
            mask = Image.open(self.mask_list[mask_nr]).convert('L')
            mask = mask.resize((image.shape[1],image.shape[0]), Image.ANTIALIAS)
             
            temp_image = Image.fromarray(image)
            temp_mask = temp_image.copy()
            temp_mask.paste(mask)
            if self.rotate:
                temp_image = temp_image.rotate(angle=angle)
                temp_mask = temp_mask.rotate(angle=angle)
                mask = mask.rotate(angle=angle)
            temp_image = Image.composite(image1=temp_image, image2=temp_mask, mask=mask)
            return {'image': numpy.array(temp_image,dtype=numpy.dtype("uint8"))}
        else:
            return {'image': image}
    

    
