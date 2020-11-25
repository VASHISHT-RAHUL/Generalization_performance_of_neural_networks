# Copyright 2020 The PGDL Competition organizers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Utilities for loading models for PGDL competition at NeurIPS 2020
# Main contributor: Yiding Jiang, July 2020

# This complexity compute a specific notion of sharpness of a function.

#30.722, 23.444
import numpy as np
import tensorflow as tf
import scipy
from scipy.sparse import *
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import LinearOperator
import timeit


def complexity(model, ds):
  t0 = timeit.default_timer()

  relu_exp = 1
  model_layers = []
  indices = []
  j = 0
  while (True):
    try:
      l_curr = model.get_layer(index=j)
      model_layers.append(l_curr)
      if len(l_curr.get_weights()) > 0:
        indices.append(j)
      # try:
      #   name = l_curr.activation.__name__
      #   if name == 'relu':
      #     indices.append(j)
      #     relu_exp = 1
      # except AttributeError:
      #   pass
      j+=1
    except ValueError:
      break

  @tf.function
  def model_output(x):
    logits = model(x)
    return logits

  indices = indices[:-1]
  tot = 0
  total_ind = 0
  for i, (x, y) in enumerate(ds.batch(1)):
    # if i!=0:
    #  print(tot/i)
    if i == 64:
      break
    for curr_ind in range(len(indices)):
      if curr_ind == 0:
        x_new = tf.identity(x)
        for j in range(indices[0]+1):
          x_new = model_layers[j](x_new)
        x_new_2 = tf.identity(x)
        tot_elems = tf.reshape(x_new_2, (-1)).shape[0]
        std = tf.sqrt(0.1/tot_elems)*tf.norm(x_new_2)
        noise = std*tf.random.normal(x_new_2.shape)
        x_new_2 = x_new_2 + noise
        for j in range(indices[0]+1):
          x_new_2 = model_layers[j](x_new_2)
        tot += 2*np.log(tf.norm(x_new_2 - x_new)/tf.norm(x_new))
        #tot += tf.norm(x_new_2 - x_new)/tf.norm(x_new)
        total_ind += 1
      else:
        for j in range(indices[curr_ind-1], indices[curr_ind] - relu_exp):
          x_new = model_layers[j+1](x_new)
        x_new_2 = tf.identity(x_new)
        for j in range(indices[curr_ind] - relu_exp, indices[curr_ind]):
          x_new = model_layers[j+1](x_new)
        tot_elems = tf.reshape(x_new_2, (-1)).shape[0]
        std = tf.sqrt(0.1 / tot_elems) * tf.norm(x_new_2)
        noise = std * tf.random.normal(x_new_2.shape)
        x_new_2 = x_new_2 + noise
        for j in range(indices[curr_ind] - relu_exp, indices[curr_ind]):
          x_new_2 = model_layers[j + 1](x_new_2)
        tot += 2 * np.log(tf.norm(x_new_2 - x_new) / tf.norm(x_new))
        #tot += (tf.norm(x_new_2 - x_new) / tf.norm(x_new))**2
        total_ind += 1

  #print(tot/total_ind)
  return tot/total_ind
