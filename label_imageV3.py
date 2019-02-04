# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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


'''

Example usage for a single image: 
python label_imageV3.py --graph=output_graph/retrained_graph.pb --labels=output_labels/retrained_labels.txt --input_layer=Placeholder --output_layer=final_result --image=samples/3.jpg --process_mode=1

Example usage for a directory: 

python label_imageV3.py --graph=output_graph/retrained_graph.pb --labels=output_labels/retrained_labels.txt --input_layer=Placeholder --output_layer=final_result --directory_name=samples/ --process_mode=2


'''



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path #for file path manipulation

import argparse
import os
import csv
import re
import time 
import numpy as np
import tensorflow as tf


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  directory_name="tensorflow/examples/label_image/data/"
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"
  process_mode = 1

#Process mode 1 = single image, 2 = multiple images from directory

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="location of the image to be processed")
  parser.add_argument("--directory_name", help="directory that contains the image to be processed")
  parser.add_argument("--process_mode", type=int, help="Interger; type of processing: 1 for single image processing, 2 for multiple images from directory (Default is 1)")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.directory_name:
    directory_name=args.directory_name
  if args.process_mode:
    process_mode=args.process_mode
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)


  # Process single image if process_mode = 1 
if process_mode == 1:

  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)


  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)

  predicted_digit=top_k[0]
  prediction_accuracy=100*(results[top_k[0]])

  print ('Digit: '+ str(predicted_digit) + ' Accuracy: '+ str(prediction_accuracy))

  with open('digit_prediction_file.csv', mode='w') as digit_prediction_file:
    prediction_writer = csv.writer(digit_prediction_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    prediction_writer.writerow([file_name, str(predicted_digit), str(prediction_accuracy)])



'''
  for i in top_k:
    print (i)  
    print(labels[i], results[i])

'''

# Process multiple images if process_mode = 2 
if process_mode == 2:
  # Start loop for each image file and print results 

  print ('================= Multiple images ============')
  string_directory_name = str(Path(directory_name) )

  csv_file_name = str(time.time())+'digit_prediction_file.csv'


  # Initialize counters for accuracy calculation
  accurate_digits=0
  total_digits=0


  # Initiatilze the CSV file and add headers
  with open(csv_file_name, mode='w',newline='') as digit_prediction_file:
    prediction_writer = csv.writer(digit_prediction_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    prediction_writer.writerow(['file_name','actual_digit', 'predicted_digit','confidence' ,'prediction_accuracy','Execution time'])


    for root, dirs, files in os.walk(string_directory_name):
        for file in files:
            file_name=os.path.join(root,file)
            label_name = os.path.split(os.path.dirname(file_name))[-1]
            label_name = re.sub(r'[^a-z0-9]+', ' ', label_name.lower())

            #Tensorflow code begins here 
            t = read_tensor_from_image_file(
            file_name,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)


            print (file_name)

            input_name = "import/" + input_layer
            output_name = "import/" + output_layer
            input_operation = graph.get_operation_by_name(input_name)
            output_operation = graph.get_operation_by_name(output_name)

            with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t
                })
            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]
            labels = load_labels(label_file)

            predicted_digit=top_k[0]
            prediction_accuracy=100*(results[top_k[0]])

            # calculate accuracy

            if str(predicted_digit).lower() == str(label_name).lower():
                accurate_digits=accurate_digits+1
                accuracy_result="True"
            else:
                accuracy_result="False"

            total_digits=total_digits+1

            # write results to the  CSV file
            prediction_writer.writerow([file_name,label_name, str(predicted_digit), str(prediction_accuracy),accuracy_result,time.ctime()])

print ('--------- results saved to CSV file--------------------')
print ('Total files tested: '+str(total_digits))
print ('Accurate predictions: '+ str(accurate_digits))
print ('Accuracy rate '+ str(accurate_digits*100/total_digits) )




                  

