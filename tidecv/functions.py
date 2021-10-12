from io import FileIO
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing, gc

import os, sys, json, ijson

def mean(arr:list):
	if len(arr) == 0:
		return 0
	return sum(arr) / len(arr)

def find_first(arr:np.array) -> int:
	""" Finds the index of the first instance of true in a vector or None if not found. """
	if len(arr) == 0:
		return None
	idx = arr.argmax()

	# Numpy argmax will return 0 if no True is found
	if idx == 0 and not arr[0]:
		return None
	
	return idx

def isiterable(x):
	try:
		iter(x)
		return True
	except:
		return False

def recursive_sum(x):
	if isinstance(x, dict):
		return sum([recursive_sum(v) for v in x.values()])
	elif isiterable(x):
		return sum([recursive_sum(v) for v in x])
	else:
		return x

def apply_messy(x:list, func):
	return [([func(y) for y in e] if isiterable(e) else func(e)) for e in x]

def apply_messy2(x:list, y:list, func):
	return [[func(i, j) for i, j in zip(a, b)] if isiterable(a) else func(a, b) for a, b in zip(x, y)]

def multi_len(x):
	try:
		return len(x)
	except TypeError:
		return 1

def unzip(l):
	return map(list, zip(*l))


def points(bbox):
	bbox = [int(x) for x in bbox]
	return (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])

def nonepack(t):
	if t is None:
		return None, None
	else:
		return t


class HiddenPrints:
	""" From https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python """

	def __enter__(self):
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		sys.stdout.close()
		sys.stdout = self._original_stdout




def toRLE(mask:object, w:int, h:int):
	"""
	Borrowed from Pycocotools:
	Convert annotation which can be polygons, uncompressed RLE to RLE.
	:return: binary mask (numpy 2D array)
	"""
	import pycocotools.mask as maskUtils

	if type(mask) == list:
		# polygon -- a single object might consist of multiple parts
		# we merge all parts into one mask rle code
		rles = maskUtils.frPyObjects(mask, h, w)
		return maskUtils.merge(rles)
	elif type(mask['counts']) == list:
		# uncompressed RLE
		return maskUtils.frPyObjects(mask, h, w)
	else:
		return mask

def maskToBoundary(mask:np.ndarray, dilation_ratio:float):
	"""
	Borrowed from LVIS:
	Convert a numpy mask to its boundary.
	Requires OpenCV.
	"""
	import cv2
	
	h, w = mask.shape
	img_diag = np.sqrt(h ** 2 + w ** 2)
	dilation = int(round(dilation_ratio * img_diag))
	if dilation < 1:
		dilation = 1
	# Pad image so mask truncated by the image border is also considered as boundary.
	new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
	kernel = np.ones((3, 3), dtype=np.uint8)
	new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
	mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]

	return mask - mask_erode


def toBoundary(rle:object, dilation_ratio:float=0.02):
	"""
	Borrowed from LVIS:
	Convert an RLE to a boundary RLE to compute Boundary IoU.
	"""
	import pycocotools.mask as maskUtils
	mask = maskUtils.decode(rle)
	boundary = maskToBoundary(mask, dilation_ratio)
	mask = maskUtils.encode(np.array(boundary[:, :, None], order='F', dtype='uint8'))[0]
	mask['counts'] = mask['counts'].decode('ascii')
	return mask

def toBoundaryAll(anns:list, dilation_ratio):
	for ann in anns:
		if ann['mask'] is not None:
			ann['mask'] = toBoundary(ann['mask'], dilation_ratio)				
	return anns

def polyToBox(poly:list):
	""" Converts a polygon in COCO lists of lists format to a bounding box in [x, y, w, h]. """

	xmin = 1e10
	xmax = -1e10
	ymin = 1e10
	ymax = -1e10

	for poly_comp in poly:
		for i in range(len(poly_comp) // 2):
			x = poly_comp[2*i + 0]
			y = poly_comp[2*i + 1]

			xmin = min(x, xmin)
			xmax = max(x, xmax)
			ymin = min(y, ymin)
			ymax = max(y, ymax)
	
	return [xmin, ymin, (xmax - xmin), (ymax - ymin)]



def chunks(lst, n):
	"""
	Yield successive n-sized chunks from lst.
	From https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
	"""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]




def _launchBoundarifyBatch(pool:multiprocessing.Pool, buffer:list, cpus:int, dilation_ratio:float):
	if len(buffer) == 0:
		return []

	anns_split = np.array_split(buffer, cpus)
	procs = []

	for ann_set in anns_split:
		p = pool.apply_async(toBoundaryAllStr, (ann_set, dilation_ratio))
		procs.append(p)
	
	return procs

def _waitBoundarifyBatch(procs:list, out_file:FileIO):
	if len(procs) == 0:
		return
	
	anns = []
	
	for p in procs:
		anns.extend(p.get())
	
	for ann in anns:
		out_file.write(ann)
		out_file.write(',')

	return anns
	
	

def toBoundaryAllStr(anns:list, dilation_ratio):
	for ann in anns:
		if ann['segmentation'] is not None:
			ann['segmentation'] = toBoundary(ann['segmentation'], dilation_ratio)
	return [json.dumps(ann) for ann in anns]


def boundarifyResults(in_path:str, out_path:str, batch_size:int=8000, cpus:int=80, dilation_ratio:float=0.02):
	""" Loads a results file and converts the masks to boundaries. """
	from tqdm import tqdm
	
	cpus = min(cpus, multiprocessing.cpu_count())
	with multiprocessing.Pool(processes=cpus) as workers:
		with open(in_path, 'r') as in_file:
			buffer = []
			procs  = []

			with open(out_path, 'w') as out_file:
				out_file.write('[')
				
				print(f'Launching {cpus} workers to process annotations in batches of {batch_size}.')

				for det in tqdm(ijson.items(in_file, 'item', use_float=True), desc='Transcribing Results'):
					if 'segmentation' not in det: det['segmentation'] = None
					if 'bbox'         not in det: det['bbox']         = None

					buffer.append(det)

					if len(buffer) >= batch_size:
						_waitBoundarifyBatch(procs, out_file) # Wait for the last batch to finish
						procs = _launchBoundarifyBatch(workers, buffer, cpus, dilation_ratio) # Start the next batch
						buffer = [] # Stagger the _waitBoundarifyBatch call so that this thread can load more data
						gc.collect() # Explicit garbage collection because this takes a lot of memory.

				# Wait for the staggered results.
				_waitBoundarifyBatch(procs, out_file)
				# In case we have a left over batch.
				procs = _launchBoundarifyBatch(workers, buffer, cpus, dilation_ratio)
				_waitBoundarifyBatch(procs, out_file)
				
				out_file.seek(out_file.tell()-1, os.SEEK_SET) # Remove the last comma
				out_file.truncate()
				out_file.write(']')
	print('Done.')
