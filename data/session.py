import fiftyone as fo

dataset = fo.zoo.load_zoo_dataset('coco-2017', split='train', max_samples=30)
session = fo.launch_app(dataset)
session.wait()