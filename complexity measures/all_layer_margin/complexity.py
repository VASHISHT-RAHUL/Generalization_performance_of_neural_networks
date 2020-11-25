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

  batch_size = 32
  layers = []
  deltas = []

  for i, (x,y) in enumerate(ds.batch(batch_size)):
    if i==1:
      break
    j = 0
    while(True):
      try:
        l_curr = model.get_layer(index=j)
        x = l_curr(x)
        layers.append(l_curr)
        if len(l_curr.get_weights()) > 0:
          deltas.append(tf.Variable(tf.zeros_like(x)))
        j += 1
      except ValueError:
        break

  lambdas = [0.0]
  all_layer_margins = None
  opt = tf.keras.optimizers.SGD(learning_rate=0.01)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
  for i, (x,y) in enumerate(ds.batch(batch_size)):
    if i==5:
      break
    pred = model(x)
    classified = tf.argmax(pred, 1, output_type=tf.dtypes.int32) == y
    del_vals = np.zeros((len(lambdas), batch_size))
    for ind, lamb in enumerate(lambdas):
      tot_rounds = 0
      while True:
        if tot_rounds > 100:
          for j in range(len(deltas)):
            deltas[j] = tf.Variable(tf.zeros_like(deltas[j]))
          break
        curr_index = 0
        x_temp = tf.identity(x)
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
          for var in deltas:
            tape.watch(var)
          for j in range(len(layers)):
            x_temp = layers[j](x_temp)
            if len(layers[j].get_weights()) > 0:
              temp_shape = np.ones(len(deltas[curr_index].shape))
              temp_shape[0]  = batch_size
              x_temp = x_temp + deltas[curr_index] * tf.reshape(tf.norm(tf.reshape(x_temp, (batch_size, -1)), axis=1, keepdims=True), tf.tuple(temp_shape))
              curr_index += 1

          new_classified = tf.argmax(x_temp, 1, output_type=tf.dtypes.int32) == y
          new_candidates = tf.math.logical_and(tf.math.logical_not(new_classified), classified)
          if tf.math.count_nonzero(new_candidates) > 0:
            classified = tf.where(new_candidates, False, classified)
            total_norm = tf.zeros((batch_size))
            for j in range(len(deltas)):
              total_norm += tf.norm(tf.reshape(deltas[j], (batch_size, -1)), axis=1)**2
            del_vals[ind, new_candidates.numpy()] = total_norm[new_candidates.numpy()].numpy()

          loss = loss_fn(y, x_temp)
          for var in deltas:
            loss = loss - lamb * (tf.norm(var) ** 2)
          # print(loss)
          loss = -loss

        grads = tape.gradient(loss, deltas)
        opt.apply_gradients(zip(grads, deltas))
        #my_var = my_var[4:8].assign(tf.zeros(4))
        for j in range(len(deltas)):
          temp = deltas[j].numpy()
          temp[tf.math.logical_not(classified).numpy()] = 0.0
          deltas[j] = tf.Variable(tf.convert_to_tensor(temp))
          #deltas[j] = deltas[j][tf.math.logical_not(classified)].assign(tf.zeros(deltas[j][tf.math.logical_not(classified)].shape))
        tot_rounds += 1

    temp = np.min(del_vals, axis=0)
    if all_layer_margins is None:
      all_layer_margins = temp[temp > 0]
    else:
      all_layer_margins = np.concatenate([all_layer_margins, temp[temp > 0]])

    #print(all_layer_margins.shape[0])
    #print(np.sqrt(np.mean(1 / np.array(all_layer_margins))))



  t2 = timeit.default_timer() - t0
  print(t2)
  return np.sqrt(np.mean(1/np.array(all_layer_margins)))

