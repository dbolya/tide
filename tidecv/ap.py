from collections import defaultdict
import numpy as np

class APDataObject:
	"""
	Stores all the information necessary to calculate the AP for one IoU and one class.
	Note: I type annotated this because why not.
	"""

	def __init__(self):
		self.data_points = {}
		self.false_negatives = set()
		self.num_gt_positives = 0
		self.curve = None

	def apply_qualifier(self, kept_preds:set, kept_gts:set) -> object:
		""" Makes a new data object where we remove the ids in the pred and gt lists. """
		obj = APDataObject()
		num_gt_removed = 0

		for pred_id in self.data_points:
			score, is_true, info = self.data_points[pred_id]

			# If the data point we kept was a true positive, there's a corresponding ground truth
			# If so, we should only add that positive if the corresponding ground truth has been kept
			if is_true and info['matched_with'] not in kept_gts:
				num_gt_removed += 1
				continue

			if pred_id in kept_preds:
				obj.data_points[pred_id] = self.data_points[pred_id]
		
		# Propogate the gt
		obj.false_negatives = self.false_negatives.intersection(kept_gts)
		num_gt_removed += (len(self.false_negatives) - len(obj.false_negatives))

		obj.num_gt_positives = self.num_gt_positives - num_gt_removed
		return obj

	def push(self, id:int, score:float, is_true:bool, info:dict={}):
		self.data_points[id] = (score, is_true, info)
	
	def push_false_negative(self, id:int):
		self.false_negatives.add(id)

	def add_gt_positives(self, num_positives:int):
		""" Call this once per image. """
		self.num_gt_positives += num_positives

	def is_empty(self) -> bool:
		return len(self.data_points) == 0 and self.num_gt_positives == 0

	def get_pr_curve(self) -> tuple:
		if self.curve is None:
			self.get_ap()
		return self.curve

	def get_ap(self) -> float:
		""" Warning: result not cached. """

		if self.num_gt_positives == 0:
			return 0

		# Sort descending by score
		data_points = list(self.data_points.values())
		data_points.sort(key=lambda x: -x[0])

		precisions = []
		recalls    = []
		num_true  = 0
		num_false = 0

		# Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
		for datum in data_points:
			# datum[1] is whether the detection a true or false positive
			if datum[1]: num_true += 1
			else: num_false += 1
			
			precision = num_true / (num_true + num_false)
			recall    = num_true / self.num_gt_positives

			precisions.append(precision)
			recalls.append(recall)

		# Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
		# Basically, remove any temporary dips from the curve.
		# At least that's what I think, idk. COCOEval did it so I do too.
		for i in range(len(precisions)-1, 0, -1):
			if precisions[i] > precisions[i-1]:
				precisions[i-1] = precisions[i]

		# Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
		resolution = 100 # Standard COCO Resoluton
		y_range = [0] * (resolution + 1) # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
		x_range = np.array([x / resolution for x in range(resolution + 1)])
		recalls = np.array(recalls)

		# I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
		# Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
		# I approximate the integral this way, because that's how COCOEval does it.
		indices = np.searchsorted(recalls, x_range, side='left')
		for bar_idx, precision_idx in enumerate(indices):
			if precision_idx < len(precisions):
				y_range[bar_idx] = precisions[precision_idx]

		self.curve = (x_range, y_range)

		# Finally compute the riemann sum to get our integral.
		# avg([precision(x) for x in 0:0.01:1])
		return sum(y_range) / len(y_range) * 100



class ClassedAPDataObject:
	""" Stores an APDataObject for each class in the dataset. """

	def __init__(self):
		self.objs = defaultdict(lambda: APDataObject())
	
	def apply_qualifier(self, pred_dict:dict, gt_dict:dict) -> object:
		ret = ClassedAPDataObject()
		
		for _class, obj in self.objs.items():
			pred_list = pred_dict[_class] if _class in pred_dict else set()
			gt_list   =   gt_dict[_class] if _class in   gt_dict else set()

			ret.objs[_class] = obj.apply_qualifier(pred_list, gt_list)
		
		return ret

	def push(self, class_:int, id:int, score:float, is_true:bool, info:dict={}):
		self.objs[class_].push(id, score, is_true, info)

	def push_false_negative(self, class_:int, id:int):
		self.objs[class_].push_false_negative(id)

	def add_gt_positives(self, class_:int, num_positives:int):
		self.objs[class_].add_gt_positives(num_positives)

	def get_mAP(self) -> float:
		aps = [x.get_ap() for x in self.objs.values() if not x.is_empty()]
		return sum(aps) / len(aps)

	def get_gt_positives(self) -> dict:
		return {k: v.num_gt_positives for k, v in self.objs.items()}

	def get_pr_curve(self, cat_id:int=None) -> tuple:
		if cat_id is None:
			# Average out the curves when using all categories
			curves = [x.get_pr_curve() for x in list(self.objs.values())]
			x_range = curves[0][0]
			y_range = [0] * len(curves[0][1])

			for x, y in curves:
				for i in range(len(y)):
					y_range[i] += y[i]
			
			for i in range(len(y_range)):
				y_range[i] /= len(curves)
		else:
			x_range, y_range = self.objs[cat_id].get_pr_curve()
		
		return x_range, y_range