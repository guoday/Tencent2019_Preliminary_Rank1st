# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Generally useful utility functions."""
from __future__ import print_function

import codecs
import collections
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
import pandas as pd
   

def hash_single_batch(batch,hparams):
    for b in batch:
        for i in range(len(b)):
            b[i]=abs(hash('key_'+str(i)+' value_'+str(b[i]))) % hparams.single_hash_num
    return batch

def hash_multi_batch(batch,hparams):
    lengths=0
    for b in batch:
        for i in range(len(b)):
            b[i]=[abs(hash('key_'+str(i)+' value_'+str(x)))% hparams.multi_hash_num for x in str(b[i]).split()] 
            lengths=max(lengths,len(b[i]))
            if len(b[i])==0:
                b[i]=[abs(hash('key_'+str(i)+' value_'+str('<pad>')))% hparams.multi_hash_num]
    batch_t=np.zeros((len(batch),len(hparams.multi_features),min(hparams.max_length,lengths)))
    weights_t=np.zeros((len(batch),len(hparams.multi_features),min(hparams.max_length,lengths)))

    
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            for k in range(min(hparams.max_length,len(batch[i][j]))):
                batch_t[i,j,k]=batch[i][j][k]
                weights_t[i,j,k]=1


    return batch_t,weights_t

def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()

def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

def print_step_info(prefix,epoch, global_step, info):
    print_out("%sepoch %d step %d lr %g loss %.6f gN %.2f, %s" %
      (prefix, epoch,global_step, info["learning_rate"],
       info["train_ppl"], info["avg_grad_norm"], time.ctime())) 
    
def print_hparams(hparams, skip_patterns=None, header=None):
  """Print hparams, can skip keys based on pattern."""
  if header: print_out("%s" % header)
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print_out("  %s=%s" % (key, str(values[key])))
