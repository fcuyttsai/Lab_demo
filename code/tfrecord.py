'''
My auto AI project in 2020
Date: 22.12.2020
Author:Richard Tsai, yuttsai@fcu.edu.tw
'''
from functools import partial
import numpy as np
import os
import logging
# import tensorflow.compat.v1 as tf  #for tensorflow 2.0
import tensorflow as tf
import scipy.signal as signal
import scipy.io as sio
import shutil
import matplotlib.pyplot as plt

# from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(decode_parse_fn)
    return dataset

def get_dataset(filenames,batchsize=128):
    dataset = load_dataset(filenames)
    # dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size= tf.data.AUTOTUNE)
    dataset = dataset.batch(batchsize)
    return dataset

def Myplots(X_, y_,title):
    """ Plot function with Inputs (values) and predictions with Predicted Labels
    :params X_: input first data as 2D-specturm
    :params y1_:  as output labels series (ie. predicts)
    :params title: graph title
    """
    ax1 = plt.subplot(211)
    plt.title(title)
    sound = np.transpose(X_)
    sound = 10. * np.log10(np.abs( sound)+1E-5)
    plt.imshow(sound, cmap='jet', interpolation='nearest', aspect='auto')
    plt.xlabel('Data series')
    plt.ylabel('Frequency features in specific FFT Length')
    plt.subplot(212, sharex = ax1)
    plt.stem(y_)
    plt.xlabel('Data series')
    plt.ylabel('Class')
    plt.xlim([0, np.size(y_)])
    plt.show()
    
def decode_parse_fnV2(example_proto):
    """
    Return decode parse data from dataset into partial.
    :params example_proto: analysis data to Tensor
    :return (Intput_data,Output_data,Name_data)
    """
    features = {"Intput_data": tf.io.FixedLenFeature((), tf.string),
                "Output_data": tf.io.FixedLenFeature((), tf.string),
                "Name_data": tf.io.FixedLenFeature((), tf.string),
                }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return {"Inputs":tf.io.decode_raw(parsed_features['Intput_data'], tf.float32),"labeled":
                        tf.io.decode_raw(parsed_features['Output_data'], tf.float32)}
    

def decode_parse_fn(example_proto):
    """
    Return decode parse data from dataset.
    :params example_proto: analysis data to Tensor
    :return (Intput_data,Output_data,Name_data)
    """
    features = {"Intput_data": tf.io.FixedLenFeature((), tf.string),
                "Output_data": tf.io.FixedLenFeature((), tf.int64),
                "Name_data": tf.io.FixedLenFeature((), tf.string),
                }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return tf.io.decode_raw(parsed_features['Intput_data'], tf.float32),tf.cast(parsed_features['Output_data'], tf.int32)#, parsed_features['Name_data']

def save_tfrecords(Intput_data, Output_data, sname, dest_file):
    """ 
    Save numpy array to TFRecord. 
    :params Intput_data: Save first array to TFRecor,each data[i] is a numpy.ndarray. (Note: Int64List or FloatList would be transfer to BytesList) 
     :params  Output_data: Save 2nd array to TFRecor,each data[i] is a numpy.ndarray. (Note: Int64List or FloatList would be transfer to BytesList)
     :params   sname:  Save 3rd array to TFRecor,each data[i] is a string array. (Note: StringList would be transfer to BytesList)
     :params   dest_file: path of the output file。 
    """
    with tf.io.TFRecordWriter(dest_file) as writer:
        for i in range(len(Intput_data)):
            # X_data_array = serialize_array(Intput_data[i,:])
            # Out_data_array = serialize_array(Output_data)
            features = tf.train.Features(
                feature={
                    "Intput_data": _bytes_feature(Intput_data[i,:]),
                    "Output_data": _bytes_feature(Output_data[i]),
                    "Name_data": _string_feature(sname.encode() )
                }
            )
            tf_example = tf.train.Example(features=features)
            serialized = tf_example.SerializeToString()
            writer.write(serialized)

def _bytes_feature(value):
    """Returns a bytes_list from a float32."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.astype(np.float32).tostring()]))

def _string_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  

def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value.astype(np.int64).tostring()])) 

def serialize_array(array):
    """ Transfer to 1D seriel array
    """
    return np.reshape(array,[1,-1])

def split_tfrecord(tfrecord_path, split_size):
    """ Tensorflow V1.x split each tfrecord data
        :params tfrecord_path: folder of tfrecord data
        :params split_size: Size of each data in items, (ie. 100 items of data)
    """
    with tf.Graph().as_default(), tf.Session() as sess:
        ds = tf.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()
        part_num = 0
        while True:
            try:
                records = sess.run(batch)
                part_path = tfrecord_path + '.{:03d}'.format(part_num)
                with tf.python_io.TFRecordWriter(part_path) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError: break
