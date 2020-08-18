# The bare-bones example from the README.
# Run coco_example.py first to get mask_rcnn_bbox.json
from tidecv import TIDE, datasets

tide = TIDE()
tide.evaluate(datasets.COCO(), datasets.COCOResult('mask_rcnn_bbox.json'), mode=TIDE.BOX)
tide.summarize()
tide.plot()
