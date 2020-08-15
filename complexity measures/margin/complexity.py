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



import numpy as np
import tensorflow as tf

def complexity(model, ds):
  

  @tf.function
  def model_output(x):
    logits = model(x)
    return logits
    
  
  margin = []
  for i, (x, y) in enumerate(ds.batch(64)):
    output = model_output(x).numpy()
    indexes = np.arange(output.shape[0]), np.argsort(output, axis=1)[:, -2]
    second_largest = output[indexes]
    margin.append(np.diag(output[:,y]) - second_largest)

  margin = np.concatenate(margin,axis=0)
  margin = np.percentile(margin[margin>0],5)
  
    
  
  return 1/(margin)**2

