import os
import time
import torch
from torchvision import transforms as T
from torchvision.datasets import CocoDetection
from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCODataset(CocoDetection):
    def __init__(self, root, annFile, n_class, resize=(416, 416)):
        super().__init__(root, annFile)

        self.n_class = n_class
        self.resize = resize    # (h, w)

        self.transform = T.Compose([
            T.PILToTensor(),
            T.Resize(resize)    # (h, w)
        ])

        # total memory needed for 416*416 trainset is < 10GB. so if you have enough memory, you might consider using cache
        self.cache = {}
    
    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            # total ~8ms
            try:
                img, target = super().__getitem__(index)    # ~3ms
            except UnidentifiedImageError:
                # if coco fails to load the img as the img's size is 0, then just load the previous one
                img, target = super().__getitem__(index-1)

            # calcaulte img size and resize scale
            width, height = img.size
            scale_x, scale_y = self.resize[1]/width, self.resize[0]/height

            # transform img
            img = self.transform(img)   # ~5ms

            # transfrom target (from COCO format(x,y,w,h) to YOLO format(cx,cy,w,h)) and rescale
            if target:
                bbox = torch.tensor([[0, obj['category_id']] + obj['bbox'] for obj in target])
                bbox[:, 2] = scale_x * (bbox[:, 2] + bbox[:, 4]/2)  # cx = x + w/2
                bbox[:, 3] = scale_y * (bbox[:, 3] + bbox[:, 5]/2)  # cy = y + h/w
            else:
                # if there's no object, just leave bbox as None
                bbox = None
            
            self.cache[index] = (img, bbox)

            return img, bbox
    
    def collate_fn(self, batch):
        start = time.time()

        imgs = []
        bboxes = []

        for i, (img, bbox) in enumerate(batch):
            imgs.append(img)

            if bbox is not None:
                bbox[:, 0] = i
                bboxes.append(bbox)

        # all images have the same size. so use stack
        # number of bboxes of each image are all different, so use cat instead with image index as index
        imgs = torch.stack(imgs)
        bboxes = torch.cat(bboxes, 0)

        end = time.time()

        print('collate_fn time:', (end - start) * 1000)
        
        return imgs, bboxes
        

if __name__ == "__main__":
    # issue 1: sudden increase in loading time at some point?
    # issue 2: UnidentifiedImageError for image 13132?

    from torch.utils.data import DataLoader

    path = 'data/coco-2017/train/'
    trainset = COCODataset(
        root=os.path.join(path, 'data'),
        annFile=os.path.join(path, 'labels.json'),
        n_class=91
    )

    trainloader = DataLoader(
        trainset, 
        batch_size=128, 
        collate_fn=trainset.collate_fn, 
        shuffle=True
    )

    start = time.time()
    for data in trainloader:
        end = time.time()
        imgs, bboxes = data
        print('loading time:', (end - start) * 1000)
        print('imgs shape:', imgs.shape)
        print('bboxes shape: ', bboxes.shape)
        # print('bboxes: ')
        # print(bboxes)
        start = time.time()
    
    