from cobaya.likelihoods.roman_kl._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_kl_interface as ci
import numpy as np

class combo_3x2pt(_cosmolike_prototype_base):
  def initialize(self):
    super(combo_3x2pt,self).initialize(probe="3x2pt")