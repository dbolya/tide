from typing import Union

class Error:
	""" A base class for all error types. """

	def fix(self) -> Union[tuple, None]:
		"""
		Returns a fixed version of the AP data point for this error or
		None if this error should be suppressed.

		Return type is:
			class:int, (score:float, is_positive:bool, info:dict)
		"""
		raise NotImplementedError

	def unfix(self) -> Union[tuple, None]:
		""" Returns the original version of this data point. """

		if hasattr(self, 'pred'):
			# If an ignored instance is an error, it's not in the data point list, so there's no "unfixed" entry
			if self.pred['used'] is None: return None
			else: return self.pred['class'], (self.pred['score'], False, self.pred['info'])
		else:
			return None
	
	def get_id(self) -> int:
		if hasattr(self, 'pred'):
			return self.pred['_id']
		elif hasattr(self, 'gt'):
			return self.gt['_id']
		else:
			return -1
	
	def get_info(self, dataset):
		info = {}
		info['type'] = self.short_name

		if hasattr(self, 'gt'):
			info['gt']   = self.gt
		if hasattr(self, 'pred'):
			info['pred'] = self.pred
		
		img_id = (self.pred if hasattr(self, 'pred') else self.gt)['image_id']
		info['all_gt'] = dataset.get(img_id)
		info['img']    = dataset.get_img(img_id)

		return info


class BestGTMatch:
	"""
	Some errors are fixed by changing false positives to true positives.
	The issue with fixing these errors naively is that you might have
	multiple errors attempting to fix the same GT. In that case, we need
	to select which error actually gets fixed, and which others just get
	suppressed (since we can only fix one error per GT).

	To address this, this class finds the prediction with the hiighest
	score and then uses that as the error to fix, while suppressing all
	other errors caused by the same GT.
	"""

	def __init__(self, pred, gt):
		self.pred = pred
		self.gt = gt

		if self.gt['used']:
			self.suppress = True
		else:
			self.suppress = False
			self.gt['usable'] = True

			score = self.pred['score']

			if not 'best_score' in self.gt:
				self.gt['best_score'] = -1

			if self.gt['best_score'] < score:
				self.gt['best_score'] = score
				self.gt['best_id'] = self.pred['_id']
		
	def fix(self):
		if self.suppress or self.gt['best_id'] != self.pred['_id']:
			return None
		else:
			return (self.pred['score'], True, self.pred['info'])
