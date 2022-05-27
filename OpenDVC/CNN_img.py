from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_compression as tfc


def MV_analysis(tensor, num_filters, M):
  """Builds the analysis transform."""

  with tf.variable_scope("MV_analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          M, (3, 3), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def MV_synthesis(tensor, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("MV_synthesis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (3, 3), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          2, (3, 3), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor


def Res_analysis(tensor, num_filters, M, reuse=False): #, channels = 3):
  """Builds the analysis transform."""

  with tf.variable_scope("analysis", reuse=reuse):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          M, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor

def Res_synthesis(tensor, num_filters, reuse=False, channels = 3):
  """Builds the synthesis transform."""

  with tf.variable_scope("synthesis", reuse=reuse):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
          channels, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor

'---------------------------MY DISCRIMINATOR --------------------------------'
slim = tf.contrib.slim

def Discriminator(x, n_hidden=128, n_layer=3, reuse=False):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        #h = slim.stack(x, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.tanh)
        h = slim.stack(x, slim.conv2d, [(32, [3, 3]), (64, [3, 3]), (128, [3, 3]), (256, [3, 3])], scope='core')
        log_d = slim.fully_connected(h, 1, activation_fn=None)
    return log_d
