# -*- coding: utf-8 -*-
# @Time    : 2020/10/2 21:26
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : logger.py
import tensorflow as tf


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

