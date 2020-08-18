# A General **T**oolbox for **I**dentifying Object **D**etection **E**rrors
```
████████╗██╗██████╗ ███████╗
╚══██╔══╝██║██╔══██╗██╔════╝
   ██║   ██║██║  ██║█████╗  
   ██║   ██║██║  ██║██╔══╝  
   ██║   ██║██████╔╝███████╗
   ╚═╝   ╚═╝╚═════╝ ╚══════╝                            
```

An easy-to-use, general toolbox to compute and evaluate the effect of object detection and instance segmentation on overall performance. This is the code for our paper: [TIDE: A General Toolbox for Identifying Object Detection Errors](https://dbolya.github.io/tide/paper.pdf).

Check out our ECCV 2020 short video for an explanation of what TIDE can do:

[![TIDE Introduction](https://img.youtube.com/vi/Pz4uggQzJXU/0.jpg)](https://www.youtube.com/watch?v=Pz4uggQzJXU)

# Installation

TIDE is available as a python package for python 3.6+ as "tidecv". To install, simply install it with pip:
```shell
pip3 install tidecv
```

# Usage
TIDE is meant as a drop-in replacement for the [COCO Evaluation toolkit](https://github.com/cocodataset/cocoapi), and getting started is easy:

```python
from tidecv import TIDE, datasets

tide = TIDE()
tide.evaluate(datasets.COCO(), datasets.COCOResult('path/to/your/results/file'), mode=TIDE.BOX) # Use TIDE.MASK for masks
tide.summarize()  # Summarize the results as tables in the console
tide.plot()       # Show a summary figure. Specify a folder and it'll output a png to that folder.
```

This prints evaluation summary tables to the console:
```
-- mask_rcnn_bbox --

bbox AP @ 50: 61.80

                         Main Errors
=============================================================
  Type      Cls      Loc     Both     Dupe      Bkg     Miss
-------------------------------------------------------------
   dAP     3.33     6.30     1.18     0.19     3.96     7.99
=============================================================

        Special Error
=============================
  Type   FalsePos   FalseNeg
-----------------------------
   dAP      16.28      15.57
=============================
```

And a summary plot for your model's errors:

![A summary plot](https://dbolya.github.io/tide/mask_rcnn_bbox_bbox_summary.png)

## Jupyter Notebook

Check out the [example notebook](https://github.com/dbolya/tide/blob/master/examples/coco_instance_segmentation.ipynb) for more details.


# Datasets
The currently supported datasets are COCO, LVIS, Pascal, and Cityscapes. More details and documentation on how to write your own database drivers coming soon!

# Citation
If you use TIDE in your project, please cite
```
@inproceedings{tide-eccv2020,
  author    = {Daniel Bolya and Sean Foley and James Hays and Judy Hoffman},
  title     = {TIDE: A General Toolbox for Identifying Object Detection Errors},
  booktitle = {ECCV},
  year      = {2020},
}
```

## Contact
For questions about our paper or code, make an issue in this github or contact [Daniel Bolya](mailto:dbolya@gatech.edu). Note that I may not respond to emails, so github issues are your best bet.
