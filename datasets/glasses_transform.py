import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import glob
import numpy

class RandomGlassesTransform:

    def __init__(self, mask_dir,
                image_type = "png",
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
                mask_nr =  random.randint(0, len(self.mask_list)-1)
            else:
                mask_nr = self.mask_nr
            mask = Image.open(self.mask_list[mask_nr]).convert('L')
            mask_background = numpy.ones((image.shape[0], image.shape[1]))
            mask_background = Image.fromarray(mask_background*255)
            mask_background = mask_background.convert('L')
            mask = mask.resize((image.shape[1],int(image.shape[0]/3)), Image.ANTIALIAS)
            mask_background.paste(mask, box=[0, int(image.shape[0]/4)])
            temp_image = Image.fromarray(image)
            temp_mask = temp_image.copy()
            temp_mask.paste(mask_background)
            if self.rotate:
                temp_image = temp_image.rotate(angle=angle)
                temp_mask = temp_mask.rotate(angle=angle)
                mask_background = mask_background.rotate(angle=angle)
            temp_image = Image.composite(image1=temp_image, image2=temp_mask, mask=mask_background)
            numpy.array(temp_image,dtype=numpy.dtype("uint8"))
            return {'image': numpy.array(temp_image,dtype=numpy.dtype("uint8"))}
        else:
            return {'image': image}

