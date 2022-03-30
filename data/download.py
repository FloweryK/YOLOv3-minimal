import fiftyone as fo

def download_coco(name, split, max_samples):
    is_done = False

    while not is_done:
        try:
            fo.zoo.load_zoo_dataset(
                name, 
                split=split, 
                max_samples=max_samples
            )
            is_done = True
        except:
            pass


if __name__ == "__main__":
    name = "coco-2017"
    download_coco(name, 'train', 40000)
    download_coco(name, 'validation', 5000)
    download_coco(name, 'test', 5000)