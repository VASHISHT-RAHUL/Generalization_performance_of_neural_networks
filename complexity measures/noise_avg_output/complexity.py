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

  tot = 0
  for i, (x, y) in enumerate(ds.batch(1)):
    # if i!=0:
    #  print(tot/i)
    if i == 256:
      break
    x_new = tf.identity(x)
    x = model(x)
    tot_elems = tf.reshape(x_new, (-1)).shape[0]
    std = tf.sqrt(0.1 / tot_elems) * tf.norm(x_new)
    noise = std * tf.random.normal(x_new.shape)
    x_new = x_new + noise
    x_new = model(x_new)
    #tot += 2 * np.log(tf.norm(x_new - x) / tf.norm(x))
    tot += (tf.norm(x_new - x) / tf.norm(x))**2
    #total_ind += 1

  #print(tot/i)
  return tot/i

# def complexity(model, ds):
#
#   t0 = timeit.default_timer()
#
#   init_weights = []
#   for i in range(len(model.initial_weights)):
#     if len(model.initial_weights[i].shape) >= 2:
#       init_weights.append(model.initial_weights[i])
#
#   input_shapes = []
#   output_shapes = []
#   for i, (x, y) in enumerate(ds.batch(64)):
#     j = 0
#     while (True):
#       try:
#         l_curr = model.get_layer(index=j)
#         input_shapes.append(x.shape)
#         x = l_curr(x)
#         output_shapes.append(x.shape)
#         j += 1
#       except ValueError:
#         break
#     if i==0:
#       break
#
#   @tf.function
#   def model_output(x):
#     logits = model(x)
#     return logits
#
#   margin = []
#   for i, (x, y) in enumerate(ds.batch(64)):
#     output = model_output(x).numpy()
#     indexes = np.arange(output.shape[0]), np.argsort(output, axis=1)[:, -2]
#     second_largest = output[indexes]
#     margin.append(np.diag(output[:, y]) - second_largest)
#
#   margin = np.concatenate(margin, axis=0)
#   margin = np.percentile(margin, 10)
#
#   t2 = timeit.default_timer() - t0
#   print(t2)
#
#   i = 0
#   depth = 0
#   frob_norm_w = []
#   spec_norm_w = []
#   frob_norm_b = []
#   frob_norm_w_diff = []
#   while(True):
#     try:
#       #print(i)
#       temp_mat = None
#       l_curr = model.get_layer(index=i)
#       temp_1 = l_curr.get_weights()
#       for j in range(len(temp_1)):
#         if len(temp_1[j].shape) >= 2:
#           temp_mat = temp_1[j]
#           ind_temp_mat = j
#           break
#       if temp_mat is not None:
#         depth += 1
#         for j in range(len(temp_1)):
#           if j != ind_temp_mat:
#             bias = temp_1[j]
#             frob_norm_b.append(np.linalg.norm(bias, 2))
#             break
#         if len(temp_mat.shape) == 2:
#           #t1 = np.concatenate([temp_mat, np.reshape(bias, (1, -1))], axis=0)
#           #log_sum += np.log(np.linalg.norm(temp_mat, 2)**2)
#           frob_norm_w.append(np.linalg.norm(temp_mat, 'fro'))
#           spec_norm_w.append(np.linalg.norm(temp_mat, 2))
#           frob_norm_w_diff.append(np.linalg.norm(temp_mat - init_weights[depth-1], 'fro'))
#           #log_sum += np.log(np.linalg.norm(t1, 2) ** 2)
#         else:
#           in_chan = input_shapes[i][3]
#           in_h = input_shapes[i][1]
#           in_w = input_shapes[i][2]
#           out_chan = output_shapes[i][3]
#           out_h = output_shapes[i][1]
#           out_w = output_shapes[i][2]
#           def mv(v):
#             #v1 = v[0:v.shape[0]-1]
#             v1 = v[0:v.shape[0]]
#             v1 = np.reshape(v1, (1, in_h, in_w, in_chan))
#             v1 = tf.convert_to_tensor(v1)
#             if l_curr.padding == 'same':
#               t1 = tf.nn.conv2d(v1, temp_mat, l_curr.strides, 'SAME')
#             else:
#               t1 = tf.nn.conv2d(v1, temp_mat, l_curr.strides, 'VALID')
#             #t1 = tf.nn.bias_add(t1, v[-1]*bias)
#             return np.reshape(t1, (-1))
#
#           def rmv(v):
#             v1 = np.reshape(v, (1, out_h, out_w, out_chan))
#             v1 = tf.convert_to_tensor(v1)
#             if l_curr.padding == 'same':
#               temp = tf.nn.conv2d_transpose(v1, temp_mat, (1, in_h, in_w, in_chan), l_curr.strides, 'SAME').numpy()
#             else:
#               temp = tf.nn.conv2d_transpose(v1, temp_mat, (1, in_h, in_w, in_chan), l_curr.strides, 'VALID').numpy()
#             #temp_dot = 0
#             #for j in range(out_h*out_w):
#             # temp_dot += np.dot(v[j*out_chan:(j+1)*out_chan], bias)
#             #return np.concatenate([np.reshape(temp, (-1)), np.array([temp_dot])])
#             return np.reshape(temp, (-1))
#
#           stride_x = int(l_curr.strides[0])
#           stride_y = int(l_curr.strides[1])
#           kernel_w = temp_mat.shape[1]
#           kernel_h = temp_mat.shape[0]
#
#           if l_curr.padding == 'valid':
#             pad_l = 0
#             pad_t = 0
#           else:
#             pad_l = int(np.floor(((np.ceil(in_w / stride_x) - 1) * stride_x + kernel_w - in_w) / 2))
#             pad_t = int(np.floor(((np.ceil(in_h / stride_y) - 1) * stride_y + kernel_h - in_h) / 2))
#
#           full_norm = np.linalg.norm(temp_mat)**2
#           full_norm_diff = np.linalg.norm(temp_mat - init_weights[depth-1])**2
#           temp_norm = 0
#           temp_norm_diff = 0
#           for j in range(out_h):
#             for k in range(out_w):
#               curr_h = stride_y*j
#               curr_w = stride_x*k
#               curr_h_fin = curr_h + kernel_h
#               curr_w_fin = curr_w + kernel_w
#               curr_h = max(pad_t-curr_h, 0)
#               curr_w = max(pad_l-curr_w, 0)
#               curr_h_fin = kernel_h - max(curr_h_fin - pad_t - in_h, 0)
#               curr_w_fin = kernel_w - max(curr_w_fin - pad_l - in_w, 0)
#               if curr_h == 0 and curr_w == 0 and curr_h_fin == kernel_h and curr_w_fin == kernel_w:
#                 temp_norm += full_norm
#                 temp_norm_diff += full_norm_diff
#               else:
#                 temp_norm += np.linalg.norm(temp_mat[curr_h:curr_h_fin, curr_w:curr_w_fin])**2
#                 temp_norm_diff += np.linalg.norm(temp_mat[curr_h:curr_h_fin, curr_w:curr_w_fin] -
#                                                  init_weights[depth-1][curr_h:curr_h_fin, curr_w:curr_w_fin]) ** 2
#
#           frob_norm_w.append(np.sqrt(temp_norm))
#           frob_norm_w_diff.append(np.sqrt(temp_norm_diff))
#           A = LinearOperator((out_chan*out_h*out_w, in_chan*in_h*in_w), matvec=mv, rmatvec=rmv)
#           s = svds(A, k=1, return_singular_vectors=False)
#           spec_norm_w.append(s[0])
#           #print(s)
#           #print(s)
#           #log_sum += np.log(s[0]**2)
#     #       stride_x = int(l_curr.strides[0])
#     #       stride_y = int(l_curr.strides[1])
#     #
#     #       if l_curr.padding == 'valid':
#     #         pad_l = 0
#     #         pad_t = 0
#     #       else:
#     #         pad_l = int(np.floor((np.ceil(in_w / stride_x) * stride_x - in_w) / 2))
#     #         pad_t = int(np.floor((np.ceil(in_h / stride_y) * stride_y - in_h) / 2))
#     #
#     #       n_col = in_chan * in_h * in_w
#     #       n_row = out_chan * out_h * out_w
#     #       kernel_w = temp_mat.shape[1]
#     #       kernel_h = temp_mat.shape[0]
#     #       #factor = 2
#     #       num = 12*np.power(10, 7)
#     #       num2 = 12*np.power(10, 5)
#     #       #temp_w = lil_matrix((n_row, n_col))
#     #       #print((n_row*kernel_h*kernel_w)/np.power(10, 5))
#     #       #print((n_row*in_chan*kernel_h*kernel_w)/np.power(10, 7))
#     #       if n_row*in_chan*kernel_h*kernel_w > num or n_row*kernel_h*kernel_w > num2:
#     #         factor = max((n_row*in_chan*kernel_h*kernel_w)/num, (n_row*kernel_h*kernel_w)/num2)
#     #       else:
#     #         factor = 1.0
#     #       print('factor: ' + str(factor))
#     #       temp_w = lil_matrix((int(n_row/factor), n_col))
#     #       for j in range(int(n_row/factor)):# * out_h * out_w):
#     #         curr_chan = j % out_chan
#     #         curr_h = int(int(j/out_chan) / out_h)
#     #         curr_w = int(j/out_chan) % out_h
#     #         input_curr_h = stride_y * curr_h
#     #         input_curr_w = stride_x * curr_w
#     #         for k in range(kernel_h):
#     #           for l in range(kernel_w):
#     #             fin_curr_h = input_curr_h + k
#     #             fin_curr_w = input_curr_w + l
#     #             if fin_curr_h >= pad_t and fin_curr_h < pad_t + in_h and fin_curr_w >= pad_l and fin_curr_w < pad_l + in_w:
#     #               curr_col = (((fin_curr_h - pad_t)* in_w) + fin_curr_w - pad_l) * in_chan
#     #               temp_w[j, curr_col:curr_col + in_chan] = temp_mat[k, l, :, curr_chan]
#     #       #print('phase 1 done')
#     #       temp_w = temp_w.tocsr()
#     #       s = svds(temp_w, k=1, return_singular_vectors=False)
#     #       print(s)
#     #       log_sum += (np.log(s[0]**2) + np.log(factor))
#     #       del(temp_w)
#       i += 1
#     except ValueError:
#       break
#
#   t3 = timeit.default_timer() - t0
#   print(t3)
#   log_sum = (1 - 1/depth)*np.sum(np.log(np.array(spec_norm_w)**2))
#   frob_spec = np.sum(np.array(frob_norm_w_diff)**2)
#   log_sum = log_sum + np.log(frob_spec) - np.log(margin**2)
#   #log_sum = log_sum - np.log(margin**2)
#   print(log_sum)
#   #log_sum = np.log(depth) + (1/depth)*(log_sum - np.log(margin**2))
#   return log_sum

