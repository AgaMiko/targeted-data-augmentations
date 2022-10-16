import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import glob
import numpy

class RandomHairTransform:
    """Rotate by one of the given angles."""

    def __init__(self, mask_dir,
                im_dir,
                image_type = "jpg",
                p=0.5,
                rotate=True,
                mask_nr='random'):
        self.mask_list =  glob.glob(mask_dir + "*."+ image_type)
        self.im_dir = im_dir
        self.len = len(self.mask_list)
        self.p = float(p)
        self.rotate = rotate
        self.mask_nr = mask_nr
    
    def __call__(self, image, **params):
        p = random.randint(0, 100)
        if p <= self.p*100:
            source_image = Image.fromarray(image)
            if self.rotate:
                angle = random.randint(0, 360)
            if self.mask_nr in ['random', None]:
                mask_nr =  random.randint(0, self.len-1)
            else:
                mask_nr = self.mask_nr
            im_path = self.im_dir + self.mask_list[mask_nr].split("/")[-1]
            mask = Image.open(self.mask_list[mask_nr]).convert('L')
            im = Image.open(im_path).convert('RGB')
            if self.rotate:
                im = im.rotate(angle=angle)
                mask = mask.rotate(angle=angle)
            temp_image = Image.composite(im, source_image, mask)
            return {'image': numpy.array(temp_image,dtype=numpy.dtype("uint8"))}
        else:
            return {'image': image}
    

    