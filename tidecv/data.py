from __future__ import annotations

from collections import defaultdict
import numpy as np

from . import functions as f

class Data():
	"""
	A class to hold ground truth or predictions data in an easy to work with format.
	Note that any time they appear, bounding boxes are [x, y, width, height] and masks
	are either a list of polygons or pycocotools RLEs.

	Also, don't mix ground truth with predictions. Keep them in separate data objects.
	
	'max_dets' specifies the maximum number of detections the model is allowed to output for a given image.
	"""


	def __init__(self, name:str, max_dets:int=100):
		self.name     = name
		self.max_dets = max_dets

		self.classes     = {}  # Maps class ID to class name 
		self.annotations = []  # Maps annotation ids to the corresponding annotation / prediction
		
		# Maps an image id to an image name and a list of annotation ids
		self.images      = defaultdict(lambda: {'name': None, 'anns': []})


	def _get_ignored_classes(self, image_id:int) -> set:
		anns = self.get(image_id)

		classes_in_image = set()
		ignored_classes  = set()

		for ann in anns:
			if ann['ignore']:
				if ann['class'] is not None and ann['bbox'] is None and ann['mask'] is None:
					ignored_classes.add(ann['class'])
			else:
				classes_in_image.add(ann['class'])
		
		return ignored_classes.difference(classes_in_image)


	def _make_default_class(self, id:int):
		""" (For internal use) Initializes a class id with a generated name. """

		if id not in self.classes:
			self.classes[id] = 'Class ' + str(id)

	def _make_default_image(self, id:int):
		if self.images[id]['name'] is None:
			self.images[id]['name'] = 'Image ' + str(id)

	def _prepare_box(self, box:object):
		return box

	def _prepare_mask(self, mask:object):
		return mask

	def _add(self, image_id:int, class_id:int, box:object=None, mask:object=None, score:float=1, ignore:bool=False):
		""" Add a data object to this collection. You should use one of the below functions instead. """
		self._make_default_class(class_id)
		self._make_default_image(image_id)
		new_id = len(self.annotations)

		self.annotations.append({
			'_id'   : new_id,
			'score' : score,
			'image' : image_id,
			'class' : class_id,
			'bbox'  : self._prepare_box(box),
			'mask'  : self._prepare_mask(mask),
			'ignore': ignore,
		})

		self.images[image_id]['anns'].append(new_id)

	def add_ground_truth(self, image_id:int, class_id:int, box:object=None, mask:object=None):
		""" Add a ground truth. If box or mask is None, this GT will be ignored for that mode. """
		self._add(image_id, class_id, box, mask)

	def add_detection(self, image_id:int, class_id:int, score:int, box:object=None, mask:object=None):
		""" Add a predicted detection. If box or mask is None, this prediction will be ignored for that mode. """
		self._add(image_id, class_id, box, mask, score=score)

	def add_ignore_region(self, image_id:int, class_id:int=None, box:object=None, mask:object=None):
		"""
		Add a region inside of which background detections should be ignored.
		You can use these to mark a region that has deliberately been left unannotated
		(e.g., if is a huge crowd of people and you don't want to annotate every single person in the crowd).

		If class_id is -1, this region will match any class. If the box / mask is None, the region will be the entire image.
		"""
		self._add(image_id, class_id, box, mask, ignore=True)

	def add_class(self, id:int, name:str):
		""" Register a class name to that class ID. """
		self.classes[id] = name
	
	def add_image(self, id:int, name:str):
		""" Register an image name/path with an image ID. """
		self.images[id]['name'] = name

	def get(self, image_id:int):
		""" Collects all the annotations / detections for that particular image. """
		return [self.annotations[x] for x in self.images[image_id]['anns']]


	def convert_to_boundary(self, dilation_ratio:float=0.02, cpus:int=1):
		"""
		Converts all of the annotation masks to be boundaries in order to use Boundary IoU.
		See this paper for more details: https://arxiv.org/abs/2103.16562

		For cpus > 1, a multiprocessing version will be used.
		"""

		if cpus <= 1:
			from tqdm import tqdm
			
			# Single threaded approach
			for ann in tqdm(self.annotations, desc='Converting to boundary'):
				if ann['mask'] is not None:
					ann['mask'] = f.toBoundary(ann['mask'], dilation_ratio)
		else:
			# Multithreaded approach
			import multiprocessing, gc

			cpus = min(cpus, multiprocessing.cpu_count())

			ignore_anns = []
			mask_anns   = []

			# Sort out the ignore regions so we can split the annotations properly
			for ann in self.annotations:
				if ann['mask'] is None:
					ignore_anns.append(ann)
				else:
					mask_anns.append(ann)
			
			# Get rid of self.annotations so we can manage memory better later
			self.annotations = []

			anns_split = np.array_split(mask_anns, cpus)
			workers = multiprocessing.Pool(processes=cpus)
			procs = []

			print(f'Launching {cpus} workers to process {len(mask_anns)} annotations.')
			for ann_set in anns_split:
				p = workers.apply_async(f.toBoundaryAll, (ann_set, dilation_ratio))
				procs.append(p)
			
			# Free the memory associated with the old annotations
			del anns_split
			del mask_anns
			gc.collect()

			print('Waiting for workers...')
			for p in procs:
				self.annotations.extend(p.get())
			
			# Add the ignore annotations back at the end.
			# I don't know if it really needs to be at the end, but I'm afraid to find out.
			self.annotations.extend(ignore_anns)
			print('Done.')




	def save(self, path:str):
		import msgpack

		out_dict = {
			'name': self.name,
			'max_dets': self.max_dets,

			'classes': self.classes,
			'annotations': self.annotations,

			'images': dict(self.images),
		}

		with open(path, 'wb') as f:
			msgpack.dump(out_dict, f)


	@staticmethod
	def load(path:str) -> Data:
		import msgpack

		with open(path, 'rb') as f:
			out_dict = msgpack.load(f, encoding='utf-8')

		data = Data(out_dict['name'], out_dict['max_dets'])

		data.classes = out_dict['classes']
		data.annotations = out_dict['annotations']
		data.images.update(out_dict['images'])

		return data

