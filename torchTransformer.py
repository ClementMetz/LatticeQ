import inspect 
import collections

import torch
import torch.nn as nn


def dict_merge(dct, merge_dct):
	""" Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
	updating only top-level keys, dict_merge recurses down into dicts nested
	to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
	``dct``.
	:param dct: dict onto which the merge is executed
	:param merge_dct: dct merged into dct
	:return: None
	source: https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
	"""
	for k, v in merge_dct.items():
		if (k in dct and isinstance(dct[k], dict)
				and isinstance(merge_dct[k], collections.Mapping)):
			dict_merge(dct[k], v)
		else:
			dct[k] = v

class TorchTransformer(nn.Module):
	"""!
	This class handle layer swap, summary, visualization of the input model
	"""
	def __init__(self):
		super(TorchTransformer, self).__init__()
		
		self._register_dict = collections.OrderedDict()

	# register class to transformer
	def register(self, origin_class, target_class):
		"""!
		This function register which class should transform to target class.		
		"""
		#print("register", origin_class, target_class)

		self._register_dict[origin_class] = target_class

		pass
	
	def trans_layers(self, model, update = True):
		"""!
		This function transform layer by layers in register dictionarys

		@param model: input model to transfer

		@param update: default is True, wether to update the paramter from the orign layer or not. 
		Note that it will update matched parameters only.

		@return transfered model
		"""
		# print("trans layer")
		if len(self._register_dict) == 0:
			print("No layer to swap")
			print("Please use register( {origin_layer}, {target_layer} ) to register layer")
			return model
		else:
			for module_name in model._modules:			
				# has children
				if len(model._modules[module_name]._modules) > 0:
					self.trans_layers(model._modules[module_name], update)
				else:
					if type(getattr(model, module_name)) in self._register_dict:
						m = model._modules[module_name]
						# use inspect.signature to know args and kwargs of __init__
						_sig = inspect.signature(type(getattr(model, module_name)))
			
						_attr_dict = getattr(model, module_name).__dict__
						_sig_dict = {}
						if type(m) == nn.Conv2d:
							_sig_dict['in_channels'] = _attr_dict['in_channels']
							_sig_dict['out_channels'] = _attr_dict['out_channels']
							_sig_dict['kernel_size'] = _attr_dict['kernel_size']
						elif type(m) == nn.Linear:
							_sig_dict['in_features'] = _attr_dict['in_features']
							_sig_dict['out_features'] = _attr_dict['out_features']
						elif type(m) == nn.MaxPool2d:
							_sig_dict['kernel_size'] = _attr_dict['kernel_size']
							_sig_dict['stride'] = _attr_dict['stride']
							_sig_dict['padding'] = _attr_dict['padding']
							_sig_dict['dilation'] = _attr_dict['dilation']
							_sig_dict['ceil_mode'] = _attr_dict['ceil_mode']
						elif type(m) == nn.AdaptiveAvgPool2d:
							_sig_dict['output_size'] = _attr_dict['output_size']
						_layer_new = self._register_dict[type(getattr(model, module_name))](**_sig_dict) 
						dict_merge(_layer_new.__dict__, _attr_dict)

						setattr(model, module_name, _layer_new)
		return model