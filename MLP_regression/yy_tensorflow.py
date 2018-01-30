# coding:utf-8
# @Time : 2018/1/8 15:01
# @Author : yuy


import argparse
import sys
import tensorflow as tf
import numpy as np
import codecs
import os
import time

FLAGS = None
TRAIN_DATA_PATH = '../data/d_train_20180102.csv'
TEST_DATA_PATH = '../data/d_test_A_20180128.csv'

# 权值的设置以及激活函数
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    # initial = tf.constant(0.1, shape=shape)
    initial = tf.random_uniform(shape, -0.1, 0.1)
    return tf.Variable(initial)


def scale_variable(shape):
    """scale_variable generates a bias variable of a given shape.
       产生一个[0.8,1.2]范围内的随机分
    """
    initial = tf.random_uniform(shape, 0.8, 1.2)
    return tf.Variable(initial)


def mean_variable(shape):
    """mean_variable generates a bias variable of a given shape."""
    initial = tf.random_uniform(shape, -0.1, 0.1)
    return tf.Variable(initial)


def symmetric_relu(x, max_value=3.0):
    return tf.maximum(tf.minimum(x, max_value), -max_value)


def read_data(fn):
        import codecs
        with codecs.open(fn, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        # 头部按,切割
        headers = lines[0].split(',')
        # print("headers: ", headers)
        # for obj in headers:
        #     print("-----", obj)
        # 按照,分割
        lines = np.array([line.split(',') for line in lines[1:]])

        feature_size = len(headers) - 4
        print(feature_size)
        genders = []
        ages = []
        values = []
        for line in lines:
            # gender = self.process_gender(line[1], aux=False)
            gender = line[1]
            # age = self.process_age(int(line[2]) if line[2] else -1, self.age_map, aux=False)
            age = int(line[2]) if line[2] else 0
            value = []
            for i in range(feature_size):
                # 从天门冬氨酸开始取
                v = line[4 + i]
                if v:
                    value.append(float(v))
                else:
                    value.append(-1)
            # if value[-1] >= 30:  # remove outlier
            #     continue
            genders.append(gender)
            ages.append(age)
            values.append(value)
            # print("value----", len(value))

        return np.array(genders), np.array(ages, dtype=np.int32), np.array(values, dtype=np.float32)


class DiabetesPredictModel:
    def __init__(self):
        self.batch_size = FLAGS.batch
        # raw train data
        self.genders, self.ages, self.values = self.read_data(TRAIN_DATA_PATH)
        # print("values--", self.values.shape[1])
        self.validate_size = FLAGS.validate
        # self.validate_size = 0
        # self.validate_size = 100
        # split raw train data
        # self.train_genders, self.train_ages, self.train_features, self.validate_genders, \
        # self.validate_ages, self.validate_features = self.split_train_validate_random()
        # part1:数据集切分
        # 训练集和验证集的切割，以及输出平均值
        self.mean_label, self.train_genders, self.train_ages, self.train_features, \
        self.validate_genders, self.validate_ages, self.validate_features = self.split_train_validate_evenly()
        # 取最后一列作为标签
        # print("--before feature size: ", self.train_features.shape)
        self.train_features, self.train_labels = self.train_features[:, :-1], self.train_features[:, -1]
        # print("before feature size: ", self.train_features.shape)
        self.mean_label = np.mean(self.train_labels)

        # part2:特征处理
        self.feature_size = self.train_features.shape[1]
        # 年龄进行映射
        self.age_array, self.age_map = self.get_age_map(self.train_ages)
        # 特征的均值，标准差，最大值
        self.feature_mean, self.feature_std, self.feature_max = self.get_mean_std_max(self.train_features)
        # 得到最大的特征
        self.feature_max_all = np.max(self.feature_max)
        # 簇的中心
        self.centers = self.get_cluster_centers(self.train_labels, clusters=512)

        # 训练集和测试集的处理
        self.train_genders, \
        self.train_ages, \
        self.train_features, \
        self.train_features_scaled, \
        self.train_feature_exists, \
        self.train_labels = self.preprocess_train_data(self.train_genders,
                                                       self.train_ages,
                                                       self.train_features,
                                                       self.train_labels)
        # print("after feature_size: ", self.feature_size)
        self.sample_indexes = np.arange(len(self.train_genders))
        # self.bucket_indexes, self.bucket_portion, self.buckets = self.split_train_data()

        # part3:对验证集进行处理
        if self.validate_size:
            self.validate_features, self.validate_labels = self.validate_features[:, :-1], self.validate_features[:, -1]
            self.mean_error = np.mean(np.square(self.validate_labels - self.mean_label))/2
            self.validate_genders, \
            self.validate_ages, \
            self.validate_features, \
            self.validate_features_scaled, \
            self.validate_feature_exists = self.preprocess_test_data(self.validate_genders, self.validate_ages, self.validate_features)

        # 产生一个3*10的大小，性别就三种可能男女未知
        self.gender_embedding_size = 10
        self.gender_embeddings = tf.Variable(
            tf.random_uniform([3, self.gender_embedding_size], -1.0, 1.0)/10)
        # 基于年龄进行embedding，向量的长度等于年龄的数目
        self.age_embedding_size = 10
        self.age_embeddings = tf.Variable(
            tf.random_uniform([len(self.age_array), self.age_embedding_size], -1.0, 1.0)/10)
        # 其他向量长度为n*7
        self.feature_embedding_size = 7
        self.feature_embeddings = tf.Variable(
            tf.random_uniform([self.feature_size, self.feature_embedding_size], -1.0, 1.0)/10)
        # 隐藏层结点数是64
        self.units1 = self.units2 = self.units3 = 64
        # 输出层的结点数是定义
        self.units4 = len(self.centers)

        # 对特征进行处理
        self.batch_gender = tf.placeholder(dtype=tf.int32, shape=[None], name='batch_gender')
        self.batch_age = tf.placeholder(dtype=tf.int32, shape=[None], name='batch_age')
        self.batch_feature = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size],
                                            name='batch_feature')
        self.batch_feature_scaled = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size],
                                                   name='batch_feature_scaled')
        self.batch_feature_exists = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size],
                                                   name='batch_feature_exists')
        self.batch_label = tf.placeholder(dtype=tf.float32, shape=[None],
                                          name='batch_label')
        self.keep_rate = tf.placeholder(dtype=tf.float32, name='dropout_keep_rate')

        # 构造网络结构
        self.output = self.build_output()
        # 得到预测结果
        self.predict, self.loss, self.loss_r = self.build_predict_loss(self.output)
        # self.learning_rate, self.train_op = self.build_train_op(self.loss)
        self.learning_rate, self.train_op = self.build_train_op(self.loss_r)

    def split_train_data(self):
        buckets = {}
        for i in range(len(self.train_labels)):
            label = int(self.train_labels[i])
            if label not in buckets:
                buckets[label] = []
            buckets[label].append(i)
        bucket_indexes = sorted(buckets.keys())
        for index in bucket_indexes:
            buckets[index] = np.array(buckets[index])
        bucket_portion = []
        for i in range(len(bucket_indexes)):
            bucket_portion.append(int(np.ceil(self.batch_size * len(buckets[bucket_indexes[i]]) / len(self.train_labels))))
        return bucket_indexes, bucket_portion, buckets

    def split_train_validate_evenly(self):
        """排序后进行训练集的切割

        :return:
        """
        indexes = sorted(np.arange(len(self.values)), key=lambda x: self.values[x][-1])
        train_indexes = []
        validate_indexes = []
        if type(self.validate_size) is int:
            self.validate_size /= len(self.values)
        # for index in indexes:
        for i in range(len(indexes)):
            index = indexes[i]
            if i == 0 or i == len(indexes) - 1:  # first and last always in train set
                train_indexes.append(index)
            elif np.random.random() < self.validate_size:
                validate_indexes.append(index)
            else:
                train_indexes.append(index)

        mean_label = np.mean(self.values[train_indexes, -1])

        self.validate_size = len(validate_indexes)
        # augmented_indexes = self.augment_train_data_large_side_5(train_indexes)
        # augmented_indexes = self.augment_train_data_large_side_5_10(train_indexes)
        augmented_indexes = self.augment_train_data_two_side(train_indexes)

        if len(validate_indexes):
            print('train data size: %d -> %d, train data mean: %.3f -> %.3f' %
                  (len(train_indexes), len(augmented_indexes), mean_label, np.mean(self.values[augmented_indexes, -1])))
            print('mean predict: %.6f -> %.6f' % (
                np.mean(np.square(mean_label - self.values[validate_indexes, -1])) / 2,
                np.mean(np.square(np.mean(self.values[augmented_indexes, -1]) - self.values[validate_indexes, -1])) / 2
            ))

        return mean_label, self.genders[augmented_indexes], self.ages[augmented_indexes], self.values[augmented_indexes], \
               self.genders[validate_indexes], self.ages[validate_indexes], self.values[validate_indexes]

    def augment_train_data_large_side_5(self, train_indexes):
        """放大最后的20%

        :param train_indexes:
        :return:
        """
        train_indexes = train_indexes + train_indexes[-len(train_indexes) // 5:]
        return train_indexes

    def augment_train_data_large_side_5_10(self, train_indexes):
        """放大前面的5%和后面的10%

        :param train_indexes:
        :return:
        """
        train_indexes = train_indexes + \
                                 train_indexes[-len(train_indexes) // 5:] + \
                                 train_indexes[-len(train_indexes) // 10:]
        return train_indexes

    def augment_train_data_two_side(self, train_indexes):
        """前后各放大10%

        :param train_indexes:
        :return:
        """
        train_indexes = train_indexes[:len(train_indexes) // 10] + \
                                 train_indexes + \
                                 train_indexes[-len(train_indexes) // 10:]
        return train_indexes

    def split_train_validate_random(self):
        if not self.validate_size:
            return self.genders, self.ages, self.values, [], [], []

        if type(self.validate_size) is float:
            self.validate_size = int(len(self.values) * self.validate_size)

        indexes = np.arange(len(self.values))
        np.random.shuffle(indexes)
        return self.genders[indexes[:-self.validate_size]], \
               self.ages[indexes[:-self.validate_size]], \
               self.values[indexes[:-self.validate_size]], \
               self.genders[indexes[-self.validate_size:]], \
               self.ages[indexes[-self.validate_size:]], \
               self.values[indexes[-self.validate_size:]]

    def build_output0(self):
        batch_gender = tf.nn.embedding_lookup(self.gender_embeddings, self.batch_gender)
        batch_age = tf.nn.embedding_lookup(self.age_embeddings, self.batch_age)
        # reshape中的[]，象征形状
        # -1代表默认可以计算出来，因为已经给了其他值，可以把-1确定出来
        batch_feature = tf.concat([
                tf.concat([tf.reshape(self.batch_feature, (-1, self.feature_size, 1))] * 5, axis=-1),
                tf.reshape(self.feature_embeddings * tf.concat([tf.reshape(self.batch_feature_exists, (-1, self.feature_size, 1))] * self.feature_embedding_size, axis=-1),
                           (-1, self.feature_size, self.feature_embedding_size))],
                axis=-1)   # 最后一个维度上连接
        batch_feature = tf.split(batch_feature, num_or_size_splits=self.feature_size, axis=1)
        layer1 = []
        extend = 32
        for i in range(self.feature_size):
            feature = tf.reshape(batch_feature[i], (-1, 5 + self.feature_embedding_size))
            layer1.append(tf.matmul(feature, weight_variable([5 + self.feature_embedding_size, extend])) + bias_variable([extend]))

        # convolution like operator

        # batch_input = tf.concat(layer1 + [batch_gender, batch_age], axis=-1)
        batch_input = tf.concat(layer1, axis=-1)
        layer_input = batch_input
        r = []
        kernels1 = 10
        self.units1 = 30
        for i in range(kernels1):
            x = tf.matmul(layer_input, weight_variable([int(layer_input.get_shape()[-1]), self.units1])) + bias_variable([self.units1])
            r.append(tf.reshape(x, (-1, self.units1, 1)))
        x = tf.concat(r, axis=-1)
        x = tf.reshape(x, (-1, self.units1 * kernels1))

        x = tf.nn.relu(x)

        x = tf.nn.batch_normalization(
            x,
            mean_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            bias_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            0.001,
            name=None
        )

        x = tf.nn.dropout(x, keep_prob=0.8)

        layer_input = x
        r = []
        kernels2 = 10
        self.units2 = 30
        for i in range(kernels2):
            x = tf.matmul(layer_input,
                          weight_variable([int(layer_input.get_shape()[-1]), self.units2])) + bias_variable(
                [self.units2])
            r.append(tf.reshape(x, (-1, self.units2, 1)))
        x = tf.concat(r, axis=-1)
        x = tf.reshape(x, (-1, self.units2 * kernels2))

        x = tf.nn.relu(x)

        x = tf.nn.batch_normalization(
            x,
            mean_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            bias_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            0.001,
            name=None
        )

        x = tf.nn.dropout(x, keep_prob=0.8)

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units3])) + bias_variable([self.units3])
        x = tf.nn.batch_normalization(
            x,
            mean_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            bias_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            0.001,
            name=None
        )
        x = tf.nn.relu(x)

        x = tf.nn.dropout(x, keep_prob=0.8)

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units4])) + bias_variable([self.units4])

        x = tf.nn.softmax(x)

        return x

    def build_output1(self):
        batch_gender = tf.nn.embedding_lookup(self.gender_embeddings, self.batch_gender)
        batch_age = tf.nn.embedding_lookup(self.age_embeddings, self.batch_age)
        batch_feature = tf.concat([
                tf.concat([
                    tf.reshape(self.batch_feature, (-1, self.feature_size, 1)),
                    tf.reshape(self.batch_feature_scaled, (-1, self.feature_size, 1)),
                    tf.reshape(self.batch_feature_scaled*self.feature_max/self.feature_max_all, (-1, self.feature_size, 1)),
                ], axis=-1),
                tf.reshape(self.feature_embeddings * tf.concat([tf.reshape(self.batch_feature_exists, (-1, self.feature_size, 1))] * self.feature_embedding_size, axis=-1), (-1, self.feature_size, self.feature_embedding_size))], axis=-1)
        batch_feature = tf.split(batch_feature, num_or_size_splits=self.feature_size, axis=1)
        layer1 = []
        extend = 32
        for i in range(self.feature_size):
            feature = tf.reshape(batch_feature[i], (-1, 3 + self.feature_embedding_size))
            layer1.append(tf.matmul(feature, weight_variable([3 + self.feature_embedding_size, extend])) + bias_variable([extend]))

        # convolution like operator

        batch_input = tf.concat(layer1 + [batch_gender, batch_age], axis=-1)
        # activate = tf.nn.relu
        activate = tf.nn.tanh
        # activate = tf.nn.sigmoid
        # activate = symmetric_relu

        skip = True if FLAGS.skip == 'true' else False

        x = batch_input

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units1])) + bias_variable([self.units1])

        x = activate(x)

        x = tf.nn.batch_normalization(
            x,
            mean_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            bias_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            0.001,
            name=None
        )

        x = tf.nn.dropout(x, keep_prob=0.8)

        if skip:
            x = tf.concat([batch_input, x], axis=-1)  # residual

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units2])) + bias_variable([self.units2])

        x = activate(x)

        x = tf.nn.batch_normalization(
            x,
            mean_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            bias_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            0.001,
            name=None
        )

        x = tf.nn.dropout(x, keep_prob=0.8)

        if skip:
            x = tf.concat([batch_input, x], axis=-1)  # residual

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units3])) + bias_variable([self.units3])

        x = activate(x)

        x = tf.nn.batch_normalization(
            x,
            mean_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            bias_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            0.001,
            name=None
        )

        x = tf.nn.dropout(x, keep_prob=0.8)

        if skip:
            x = tf.concat([batch_input, x], axis=-1)  # residual

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units4])) + bias_variable([self.units4])

        x = tf.nn.softmax(x)

        return x

    def build_output2(self):
        batch_gender = tf.nn.embedding_lookup(self.gender_embeddings, self.batch_gender)
        batch_age = tf.nn.embedding_lookup(self.age_embeddings, self.batch_age)
        batch_feature = tf.concat([
                tf.concat([
                    tf.reshape(self.batch_feature, (-1, self.feature_size, 1)),
                    tf.reshape(self.batch_feature_scaled, (-1, self.feature_size, 1)),
                    tf.reshape(self.batch_feature_scaled*self.feature_max/self.feature_max_all, (-1, self.feature_size, 1)),
                ], axis=-1),
                tf.reshape(self.feature_embeddings * tf.concat([tf.reshape(self.batch_feature_exists, (-1, self.feature_size, 1))] * self.feature_embedding_size, axis=-1), (-1, self.feature_size, self.feature_embedding_size))], axis=-1)
        batch_feature = tf.split(batch_feature, num_or_size_splits=self.feature_size, axis=1)
        layer = []
        extend = 10
        for i in range(self.feature_size):
            feature = tf.reshape(batch_feature[i], (-1, 3 + self.feature_embedding_size))
            layer.append(tf.matmul(feature, weight_variable([3 + self.feature_embedding_size, extend])) + bias_variable([extend]))

        # convolution like operator

        batch_input = tf.concat(layer + [batch_gender, batch_age], axis=-1)
        # activate = tf.nn.relu
        activate = tf.nn.tanh
        # activate = tf.nn.sigmoid
        # activate = symmetric_relu

        skip = True if FLAGS.skip == 'true' else False

        x = batch_input

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units1])) + bias_variable([self.units1])

        x = activate(x)

        x = tf.nn.batch_normalization(
            x,
            mean_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            bias_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            0.001,
            name=None
        )

        x = tf.nn.dropout(x, keep_prob=0.8)

        if skip:
            layer = []
            extend = 10
            for i in range(self.feature_size):
                feature = tf.reshape(batch_feature[i], (-1, 3 + self.feature_embedding_size))
                layer.append(
                    tf.matmul(feature, weight_variable([3 + self.feature_embedding_size, extend])) + bias_variable(
                        [extend]))
            x = tf.concat(layer + [x], axis=-1)  # residual

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units2])) + bias_variable([self.units2])

        x = activate(x)

        x = tf.nn.batch_normalization(
            x,
            mean_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            bias_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            0.001,
            name=None
        )

        x = tf.nn.dropout(x, keep_prob=0.8)

        if skip:
            layer = []
            extend = 10
            for i in range(self.feature_size):
                feature = tf.reshape(batch_feature[i], (-1, 3 + self.feature_embedding_size))
                layer.append(
                    tf.matmul(feature, weight_variable([3 + self.feature_embedding_size, extend])) + bias_variable(
                        [extend]))
            x = tf.concat(layer + [x], axis=-1)  # residual

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units3])) + bias_variable([self.units3])

        x = activate(x)

        x = tf.nn.batch_normalization(
            x,
            mean_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            bias_variable([int(x.get_shape()[-1])]),
            scale_variable([int(x.get_shape()[-1])]),
            0.001,
            name=None
        )

        x = tf.nn.dropout(x, keep_prob=0.8)

        if skip:
            layer = []
            extend = 10
            for i in range(self.feature_size):
                feature = tf.reshape(batch_feature[i], (-1, 3 + self.feature_embedding_size))
                layer.append(
                    tf.matmul(feature, weight_variable([3 + self.feature_embedding_size, extend])) + bias_variable(
                        [extend]))
            x = tf.concat(layer + [x], axis=-1)  # residual

        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units4])) + bias_variable([self.units4])

        x = tf.nn.softmax(x)

        return x

    def select_activate(self):
        activate_map = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'srelu': symmetric_relu
        }
        return activate_map[FLAGS.activate]

    def build_output(self):
        batch_gender = tf.nn.embedding_lookup(self.gender_embeddings, self.batch_gender)
        print("gender: ", batch_gender.shape)
        batch_age = tf.nn.embedding_lookup(self.age_embeddings, self.batch_age)
        print("age: ", batch_age.shape)
        batch_feature = tf.concat([
                tf.concat([
                    tf.reshape(self.batch_feature, (-1, self.feature_size, 1)),
                    tf.reshape(self.batch_feature_scaled, (-1, self.feature_size, 1)),
                    tf.reshape(self.batch_feature_scaled*self.feature_max/self.feature_max_all, (-1, self.feature_size, 1)),
                ], axis=-1),
                tf.reshape(self.feature_embeddings * tf.concat([tf.reshape(self.batch_feature_exists, (-1, self.feature_size, 1))]
                           * self.feature_embedding_size, axis=-1), (-1, self.feature_size, self.feature_embedding_size))], axis=-1)
        print("feature: ", batch_feature.shape)
        batch_input = tf.concat([
            tf.reshape(batch_gender, (-1, 1, self.gender_embedding_size)),
            tf.reshape(batch_age, (-1, 1, self.age_embedding_size)),
            batch_feature
        ], axis=-2)

        x = None

        # 网络的深度
        depth = FLAGS.depth
        activate = self.select_activate()

        skip = True if FLAGS.skip == 'true' else False
        # 隐藏节点的个数
        hidden_units = FLAGS.hidden

        for _ in range(depth):
            if x is None or skip:
                # 维数好像不对，这里改一下，从39改到40试一试！！！！！
                # 根据input得到output
                o = tf.reduce_sum(batch_input * weight_variable([1, 39, 10]), axis=-1) + bias_variable([39])
                # o = tf.reduce_sum(batch_input * weight_variable([1, 40, 10]), axis=-1) + bias_variable([40])
                # 对于output进行激活
                o = activate(o)
            # 输入层，输出就等于输入
            if x is None:
                x = o
            # skip=True，每次都保持原始特征
            elif skip:
                x = tf.concat([x, o], axis=-1)
            # 再得到权重矩阵的偏置trans
            x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), hidden_units])) + bias_variable([hidden_units])
            # 又激活
            x = activate(x)

            x = tf.nn.batch_normalization(
                x,
                mean_variable([int(x.get_shape()[-1])]),
                scale_variable([int(x.get_shape()[-1])]),
                bias_variable([int(x.get_shape()[-1])]),
                scale_variable([int(x.get_shape()[-1])]),
                0.001,
                name=None
            )

            x = tf.nn.dropout(x, keep_prob=self.keep_rate)
        # 最后一层输出层
        x = tf.matmul(x, weight_variable([int(x.get_shape()[-1]), self.units4])) + bias_variable([self.units4])
        # 再使用softmax函数进行输出
        x = tf.nn.softmax(x)

        return x

    def build_predict_loss(self, x):
        predict = tf.matmul(x, self.centers)
        predict = tf.reshape(predict, shape=[-1])
        # loss = tf.losses.mean_squared_error(labels=self.batch_label, predictions=predict)
        loss = tf.reduce_mean(tf.square(self.batch_label - predict)) / 2

        # loss_r = self.get_weighted_loss0(predict)  # weighted loss
        # loss_r = self.get_weighted_loss0_r(predict)  # weighted loss
        # loss_r = self.get_weighted_loss1_r(predict)  # weighted loss
        # loss_r = self.get_weighted_loss2(predict)  # weighted loss
        # loss_r = self.get_weighted_loss2_sign(predict, rate=0.1)  # weighted loss
        # loss_r = self.get_weighted_loss2_r(predict)  # weighted loss
        # loss_r = self.get_weighted_loss_log(predict)  # weighted loss
        loss_r = loss  # plain loss

        # l2 loss
        # loss_r += self.get_l2_loss() * 0.00001

        return predict, loss, loss_r

    def get_l2_loss(self):
        """
        l2 loss for regularization
        :return:
        """
        l2_loss = 0
        for v in tf.trainable_variables():
            l2_loss += tf.nn.l2_loss(v)
        return l2_loss

    def get_weighted_loss1(self, predict):
        weight = self.feature_size + 2 + 1 - (
            tf.cast(tf.not_equal(self.batch_gender, 0), dtype=tf.float32) + \
            tf.cast(tf.not_equal(self.batch_age, 0), dtype=tf.float32) + \
            tf.count_nonzero(self.batch_feature_exists, axis=[-1], dtype=tf.float32))
        loss_r = tf.square(self.batch_label - predict)
        loss_r /= weight
        loss_r = tf.reduce_mean(loss_r)
        return loss_r

    def get_weighted_loss1_r(self, predict):
        weight = self.feature_size + 2 + 1 - (
            tf.cast(tf.not_equal(self.batch_gender, 0), dtype=tf.float32) + \
            tf.cast(tf.not_equal(self.batch_age, 0), dtype=tf.float32) + \
            tf.count_nonzero(self.batch_feature_exists, axis=[-1], dtype=tf.float32))
        loss_r = tf.square(self.batch_label - predict)
        loss_r *= weight
        loss_r = tf.reduce_mean(loss_r)
        return loss_r

    def get_weighted_loss0(self, predict):
        """
        weight according to information completeness
        :param predict:
        :return:
        """
        weight = (tf.cast(tf.not_equal(self.batch_gender, 0), dtype=tf.float32) + \
                  tf.cast(tf.not_equal(self.batch_age, 0), dtype=tf.float32) + \
                  tf.count_nonzero(self.batch_feature_exists, axis=[-1], dtype=tf.float32)) / (self.feature_size + 2)
        weight **= 2
        loss_r = tf.square(self.batch_label - predict)
        loss_r *= weight
        loss_r = tf.reduce_mean(loss_r)
        return loss_r

    def get_weighted_loss0_r(self, predict):
        weight = (1 + tf.cast(tf.not_equal(self.batch_gender, 0), dtype=tf.float32) + \
                  tf.cast(tf.not_equal(self.batch_age, 0), dtype=tf.float32) + \
                  tf.count_nonzero(self.batch_feature_exists, axis=[-1], dtype=tf.float32)) / (self.feature_size + 2)
        weight **= 2
        loss_r = tf.square(self.batch_label - predict)
        loss_r /= weight
        loss_r = tf.reduce_mean(loss_r)
        return loss_r

    def get_weighted_loss2(self, predict):
        """
        weight according to label variation
        :param predict:
        :return:
        """
        weight = np.abs(self.batch_label - self.mean_label) * 0.1 + 1
        loss_r = tf.square(self.batch_label - predict)
        loss_r *= weight
        loss_r = tf.reduce_mean(loss_r)
        return loss_r

    def get_weighted_loss2_r(self, predict):
        weight = np.abs(self.batch_label - self.mean_label) * 0.1 + 1
        loss_r = tf.square(self.batch_label - predict)
        loss_r /= weight
        loss_r = tf.reduce_mean(loss_r)
        return loss_r

    def get_weighted_loss2_sign(self, predict, rate=0.1):
        weight = tf.maximum(0., self.batch_label - 6.) * rate + 1
        loss_r = tf.square(self.batch_label - predict)
        loss_r *= weight
        loss_r = tf.reduce_mean(loss_r)
        return loss_r

    def get_weighted_loss_log(self, predict):
        weight = self.batch_label * 0.1
        loss_r = tf.log(tf.square(self.batch_label - predict) + 1)
        loss_r /= weight
        loss_r = tf.reduce_mean(loss_r)
        return loss_r

    def get_cluster_centers(self, labels, clusters):
        # min_v = min(labels)
        # max_v = max(labels)
        # # mean = (min_v + max_v) / 2
        # mean = np.mean(labels)
        # min_v += (min_v - mean) * 0.2
        # max_v += (max_v - mean) * 0.2
        # 转换为列向量
        min_v, max_v = 0, 40
        # 血糖的范围是[0,40]，切分为512个区间，并且把行向量转换为列向量
        centers = np.linspace(min_v, max_v, clusters, dtype=np.float32)
        centers.shape = (-1, 1)
        return centers

    def get_mean_std_zero(self, values):
        # 得到每个特征的最大值
        max_e = np.ones(self.feature_size, np.float32)
        for sample in values:
            for i in range(self.feature_size):
                if sample[i] >= 0:
                    max_e[i] = max(max_e[i], sample[i])
        return np.zeros(self.feature_size, np.float32), max_e

    def get_mean_std_max(self, values):
        total_sum = np.zeros(self.feature_size, np.float32)
        total_count = np.zeros(self.feature_size, np.float32)
        feature_max = np.zeros(self.feature_size, np.float32)
        for sample in values:
            for i in range(self.feature_size):
                if sample[i] >= 0:
                    total_sum[i] += sample[i]
                    total_count[i] += 1
                    feature_max[i] = max(feature_max[i], sample[i])
        mean = total_sum / total_count

        total_sum = np.zeros(self.feature_size, np.float32)
        total_count = np.zeros(self.feature_size, np.float32)
        for sample in values:
            for i in range(self.feature_size):
                if sample[i] >= 0:
                    total_sum[i] += (sample[i] - mean[i]) ** 2
                    total_count[i] += 1
        std = np.sqrt(total_sum / total_count)

        return mean, std, feature_max

    def preprocess_train_data(self, genders, ages, values, labels):
        sample_genders = []
        sample_ages = []
        # 归一化之后的特征
        sample_features_normalized = []
        # 加噪之后的特征
        sample_features_scaled = []
        # 特征是否存在
        sample_feature_exists = []
        sample_labels = []
        for i in range(len(genders)):
            gender = self.process_gender(genders[i], aux=False)
            age = self.process_age(ages[i], aux=False)
            existing_features = np.ones(self.feature_size, dtype=np.float32)
            scaled_features = np.array(values[i])
            normalized_features = np.array(values[i])
            for j in range(self.feature_size):
                if values[i][j] < 0:
                    normalized_features[j] = self.feature_mean[j]
                    scaled_features[j] = 0
                    existing_features[j] = 0

            sample_genders.append(gender)
            sample_ages.append(age)
            sample_features_normalized.append(normalized_features)
            sample_features_scaled.append(scaled_features)
            sample_feature_exists.append(existing_features)
            sample_labels.append(labels[i])

        genders = np.array(sample_genders, dtype=np.int32)
        ages = np.array(sample_ages, dtype=np.int32)
        sample_features_normalized = np.array(sample_features_normalized, dtype=np.float32)
        sample_features_scaled = np.array(sample_features_scaled, dtype=np.float32)
        existing_features = np.array(sample_feature_exists, dtype=np.float32)
        labels = np.array(sample_labels, dtype=np.float32)

        return genders, ages, (sample_features_normalized - self.feature_mean) / self.feature_std, \
               sample_features_scaled / self.feature_max, existing_features, labels

    def preprocess_test_data(self, genders, ages, values):
        sample_genders = []
        sample_ages = []
        sample_features_normalized = []
        sample_features_scaled = []
        sample_feature_exists = []
        for i in range(len(genders)):
            gender = self.process_gender(genders[i], aux=False)
            age = self.process_age(ages[i], aux=False)
            existing_features = np.ones(self.feature_size, dtype=np.float32)
            scaled_features = np.array(values[i])
            normalized_features = np.array(values[i])
            for j in range(self.feature_size):
                # -1的话代表该特征缺失
                # 采用该列的平均值进行替代
                # 第i个记录的第j个特征
                if values[i][j] < 0:
                    normalized_features[j] = self.feature_mean[j]
                    scaled_features[j] = 0
                    # 存在位标记为0
                    existing_features[j] = 0

            sample_genders.append(gender)
            sample_ages.append(age)
            sample_features_normalized.append(normalized_features)
            sample_features_scaled.append(scaled_features)
            sample_feature_exists.append(existing_features)

        genders = np.array(sample_genders, dtype=np.int32)
        ages = np.array(sample_ages, dtype=np.int32)
        sample_features_normalized = np.array(sample_features_normalized, dtype=np.float32)
        sample_features_scaled = np.array(sample_features_scaled, dtype=np.float32)
        existing_features = np.array(sample_feature_exists, dtype=np.float32)

        # 减去均值除以标准差，进行标准化处理
        # 特征归一化，除以最大值
        # 特征是否存在的标志位
        return genders, ages, (sample_features_normalized - self.feature_mean) / self.feature_std, \
               sample_features_scaled / self.feature_max, existing_features

    def process_gender(self, gender, aux=True):
        if gender == '男':
            return 1 if not aux else [0, 1]
        elif gender == '女':
            return 2 if not aux else [0, 2]
        else:
            return 0 if not aux else [0]

    def process_age(self, age, aux=True):
        age_map = self.age_map
        if age in age_map and age_map[age] != 0:
            return age_map[age] if not aux else [0, age_map[age]]
        else:
            return 0 if not aux else [0]

    def get_age_map(self, ages):
        age_map = {}
        for age in ages:
            if age in age_map:
                age_map[age] += 1
            else:
                age_map[age] = 1
        ages = sorted(list(age_map.items()), key=lambda x: x[0])
        age_array = [-1]
        age_map = {-1: 0}
        for age, count in ages:
            if count >= 5:
                age_array.append(age)
                age_map[age] = len(age_map)
        return age_array, age_map

    @staticmethod
    def build_train_op(loss):
        learning_rate = tf.placeholder(dtype=tf.float32)
        optimizers = {
            'sgd':          tf.train.GradientDescentOptimizer,
            'adagrad':      tf.train.AdagradOptimizer,
            'adadelta':     tf.train.AdadeltaOptimizer,
            'adam':         tf.train.AdamOptimizer,
            'momentum':     tf.train.MomentumOptimizer,
            'rmsp':         tf.train.RMSPropOptimizer
        }
        optimizer_class = optimizers[FLAGS.optimizer]
        if FLAGS.optimizer == 'adam':
            optimizer = optimizer_class(learning_rate, epsilon=1e-2)
        elif FLAGS.optimizer == 'momentum':
            optimizer = optimizer_class(learning_rate, 0.9)
        else:
            optimizer = optimizer_class(learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimizer = tf.train.AdagradOptimizer(learning_rate)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-2)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # optimizer = tf.contrib.opt.NadamOptimizer(learning_rate)
        # optimizer = tf.train.FtrlOptimizer(learning_rate)
        # gvs = optimizer.compute_gradients(loss)
        # capped_gvs = [(tf.clip_by_value(grad, -5.0/10, 5.0/10), var) if grad is not None else (grad, var) for grad, var in gvs]
        # train_op = optimizer.apply_gradients(capped_gvs)
        train_op = optimizer.minimize(loss)
        return learning_rate, train_op

    def train(self, sess, epochs=1):
        print('parameters: %d' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        sess.run(tf.global_variables_initializer())

        if not FLAGS.optimizer == 'adadelta':
            learning_rate = 0.0001
        else:
            learning_rate = 1e-1
        one_pass = int(np.ceil(len(self.train_labels) / self.batch_size))
        steps = one_pass * epochs   # 迭代次数
        decay = 10**(1/epochs)
        min_loss = 10
        for epoch in range(epochs):
            for i in range(one_pass):
                # 得到原始特征，标准化之后的特征，特征是否存在
                # 特征的y值
                batch_gender, batch_age, batch_feature, batch_feature_scaled, batch_feature_exists, batch_label = \
                    self.sample(self.batch_size * i, self.batch_size * (i+1))
                # batch_gender, batch_age, batch_feature, batch_feature_exists, batch_label = \
                #     self.sample_evenly()
                predict, loss, loss_r, _ = sess.run([self.predict, self.loss, self.loss_r, self.train_op], feed_dict={
                        self.batch_gender: batch_gender,
                        self.batch_age: batch_age,
                        self.batch_feature: batch_feature,
                        self.batch_feature_scaled: batch_feature_scaled,
                        self.batch_feature_exists: batch_feature_exists,
                        self.batch_label: batch_label,
                        self.keep_rate: FLAGS.dropout,
                        self.learning_rate: learning_rate,
                })
                if not self.validate_size:
                # if True:
                    print('%d / %d, loss=%.6f, loss_r=%.6f' % (one_pass * epoch + i + 1, steps, loss, loss_r))
                if np.isnan(loss):
                    exit(0)
                if self.validate_size and (i+1) == one_pass and (epoch+1) % 1 == 0:
                    loss, loss_r = self.validate(sess)
                    print('%d / %d, validate_loss=%.6f/%.6f/%.6f, validate_loss_r=%.6f' %
                          (one_pass * epoch + i + 1, steps, loss, self.mean_error, loss / self.mean_error,  loss_r))
                    # if loss / self.mean_error < FLAGS.threshold:
                    #     return
                    min_loss = min(min_loss, loss)
                # for obj in predict:
                #     print(obj)
            learning_rate /= decay

    def validate(self, sess, print_predict=False):
        start = 0
        total_loss = 0
        total_loss_r = 0
        # import time
        # t = str(time.time())
        # with codecs.open('%s.txt' % t, 'w', encoding='utf-8') as f:
        if True:
            while start < self.validate_size:
                # 取出当前的块
                end = min(start + self.batch_size, self.validate_size)
                # 通过对数据集采样进行
                # 提取出其中的性别，年龄，别的特征，加噪标签等
                batch_gender, batch_age, batch_feature, batch_feature_scaled, batch_feature_exists, batch_label = \
                    self.validate_sample(start, end)
                # for gender in batch_gender:
                #     print(gender, file=f)
                # for age in batch_age:
                #     print(age, file=f)
                # for feature in batch_feature:
                #     print(feature, file=f)
                # for scaled_feature in batch_feature_scaled:
                #     print(scaled_feature, file=f)
                # for feature_exists in batch_feature_exists:
                #     print(feature_exists, file=f)
                # for label in batch_label:
                #     print(label, file=f)
                predict, loss, loss_r = sess.run([self.predict, self.loss, self.loss_r], feed_dict={
                    self.batch_gender: batch_gender,
                    self.batch_age: batch_age,
                    self.batch_feature: batch_feature,
                    self.batch_feature_scaled: batch_feature_scaled,
                    self.batch_feature_exists: batch_feature_exists,
                    self.batch_label: batch_label,
                    self.keep_rate: 1.0,   # 保留比
                    # self.learning_rate: learning_rate
                })
                total_loss += loss * (end - start)
                total_loss_r += loss_r * (end - start)
                start = end
                # 输出验证集的预测结果
                # if print_predict:
                #     for p, l in zip(predict, batch_label):
                #         print('%.3f\t%.3f' % (p, l))
        return total_loss / end, total_loss_r / end

    def sample_evenly(self):
        """特征集的涂抹处理，1/3保持原样，1/3随机抹，1/3按照确实比例进行抹除

        :return:
        """
        batch_gender = []
        batch_ages = []
        batch_features = []
        batch_feature_exists = []
        batch_labels = []

        for i in range(len(self.bucket_indexes)):
            index = self.bucket_indexes[i]
            count = self.bucket_portion[i]
            indexes = np.unique(np.random.randint(len(self.buckets[index]), size=int(count*1.2)))
            for j in self.buckets[index][indexes]:
                batch_gender.append(self.train_genders[j])
                batch_ages.append(self.train_ages[j])
                batch_features.append(self.train_features[j])
                batch_feature_exists.append(self.train_feature_exists[j])
                batch_labels.append(self.train_labels[j])
        batch_gender = np.asarray(batch_gender)
        batch_ages = np.asarray(batch_ages)
        batch_features = np.array(batch_features)
        batch_feature_exists = np.array(batch_feature_exists)
        batch_labels = np.asarray(batch_labels)

        for i in range(len(batch_features)):
            select = np.random.random()
            if select < 0.33333:
                augment_features = self.augment_features_random
            elif select < 0.66666:
                augment_features = self.augment_features_keep
            else:
                augment_features = self.augment_features_dummy
            augment_features(batch_features[i], batch_feature_exists[i])

        return batch_gender, batch_ages, batch_features, batch_feature_exists, batch_labels

    def sample(self, start, end):
        if start == 0:
            np.random.shuffle(self.sample_indexes)
        indexes = self.sample_indexes
        batch_gender = self.train_genders[indexes[start:end]]
        batch_ages = self.train_ages[indexes[start:end]]
        batch_features = self.train_features[indexes[start:end]]
        batch_features_scaled = self.train_features_scaled[indexes[start:end]]
        batch_feature_exists = self.train_feature_exists[indexes[start:end]]
        batch_labels = self.train_labels[indexes[start:end]]

        batch_gender = np.array(batch_gender)
        batch_ages = np.array(batch_ages)
        batch_features = np.array(batch_features)
        batch_features_scaled = np.array(batch_features_scaled)
        batch_feature_exists = np.array(batch_feature_exists)

        for i in range(len(batch_features)):
            if np.random.random() < 0.5:
                batch_gender[i] = 0
            if np.random.random() < 0.5:
                batch_ages[i] = 0
            select = np.random.random()
            if select < 0.33333:
                augment_features = self.augment_features_random
            elif select < 0.66666:
                augment_features = self.augment_features_keep
            else:
                augment_features = self.augment_features_dummy
            augment_features(batch_features[i], batch_features_scaled[i], batch_feature_exists[i])
            self.feature_jittering(batch_features[i], batch_features_scaled[i], batch_feature_exists[i])

        return batch_gender, batch_ages, batch_features, batch_features_scaled, batch_feature_exists, batch_labels

    def augment_features_dummy(self, features, features_scaled, feature_exists):
        pass

    def augment_features_random(self, features, features_scaled, feature_exists):
        size = np.random.randint(self.feature_size)
        # indexes = np.random.randint(self.feature_size, size=size)
        # 以前抹去1/3的特征
        # 改成抹去1/10的特征
        indexes = np.random.randint(self.feature_size, size=size//3)
        # indexes = np.random.randint(self.feature_size, size=size//5)
        features[indexes] = 0
        features_scaled[indexes] = 0
        feature_exists[indexes] = 0

    def augment_features_keep(self, features, features_scaled, feature_exists):
        """37个feature的缺失情况

        :param features:
        :param features_scaled:
        :param feature_exists:
        :return:
        """
        keep = np.array([0.815, 0.815, 0.815, 0.815, 0.815, 0.815, 0.815, 0.815, \
                         0.824, 0.824, 0.824, 0.824, \
                         0.806, 0.806, 0.806, \
                         0.169, 0.169, 0.169, 0.169, 0.169, \
                         0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995, \
                         0.994, 0.994, 0.994, \
                         0.995, 0.995, 0.995, 0.995, 0.995
        ])
        # print("----keep size: ", len(keep))
        # keep = (1 - keep) * 0.5 + keep
        keep *= 0.9
        for j in range(self.feature_size):
            if np.random.random() >= keep[j]:
                features[j] = 0
                features_scaled[j] = 0
                feature_exists[j] = 0

    def feature_jittering(self, features, features_scaled, feature_exists):
        for j in range(self.feature_size):
            if not feature_exists[j]:
                continue
            r = (np.random.random() - 0.5) / 50 + 1
            origin = features_scaled[j] * self.feature_max[j]
            features[j] = (origin * r - self.feature_mean[j]) / self.feature_std[j]
            features_scaled[j] *= r

    def feature_jittering0(self, features, features_scaled, feature_exists):
        """特征加噪，把特征的取值乘以0.99-1.01的系数

        :param features:
        :param features_scaled:
        :param feature_exists:
        :return:
        """
        for j in range(self.feature_size):
            if not feature_exists[j]:
                continue
            r = np.random.random()
            a = np.random.randint(0, 4)
            # 乘以小的数字
            if r < 0.33333:
                list = [0.98, 0.985, 0.99, 0.995]
                origin = features_scaled[j] * self.feature_max[j] * 0.99
                features[j] = (origin - self.feature_mean[j]) / self.feature_std[j]
                features_scaled[j] *= 0.99
            # 乘以大的数字
            elif r < 0.66666:
                list = [1.005, 1.01, 1.015, 1.02]
                origin = features_scaled[j] * self.feature_max[j] * 1.01
                features[j] = (origin - self.feature_mean[j]) / self.feature_std[j]
                features_scaled[j] *= 1.01
            else:
                pass

    def validate_sample(self, start, end):
        """验证集采样

        :param start:
        :param end:
        :return:
        """
        # 提取特征
        batch_gender = self.validate_genders[start:end]
        batch_ages = self.validate_ages[start:end]
        batch_features = self.validate_features[start:end]
        # 加噪提取y
        batch_features_scaled = self.validate_features_scaled[start:end]
        batch_feature_exists = self.validate_feature_exists[start:end]
        batch_labels = self.validate_labels[start:end]
        return batch_gender, batch_ages, batch_features, batch_features_scaled, batch_feature_exists, batch_labels

    def test(self, sess, i):
        """测试集的保持比例100

        :param sess:
        :param i:
        :return:
        """
        genders, ages, normalized_features = self.read_data(TEST_DATA_PATH)
        genders, ages, normalized_features, scaled_features, feature_exists = self.preprocess_test_data(genders, ages, normalized_features)
        test_size = len(genders)
        start = 0
        # with codecs.open("result.csv", "w", encoding="utf-8") as f:
        # with codecs.open(FLAGS.save, 'w', encoding='utf-8') as f:
        total_predict = []
        while start < test_size:
            end = min(start + self.batch_size, test_size)
            batch_gender, batch_age, batch_feature, batch_feature_scaled, batch_feature_exists = \
                genders[start:end], ages[start:end], normalized_features[start:end], scaled_features[start:end], feature_exists[start:end]
            predict, = sess.run([self.predict], feed_dict={
                self.batch_gender: batch_gender,
                self.batch_age: batch_age,
                self.batch_feature: batch_feature,
                self.batch_feature_scaled: batch_feature_scaled,
                self.batch_feature_exists: batch_feature_exists,
                self.keep_rate: 1.0
            })
            total_predict.extend(list(predict))
            # for v in predict:
            # print('%.3f' % v, file=f)
            # f.write("%.3f" %v)
            start = end
        import pandas as pd
        predict = pd.DataFrame(total_predict)
        predict.to_csv("2018_01_26_dep8_round300_" + str(i) + ".csv", header=None, index=False, float_format='%.3f')
        return predict

    def read_data(self, fn):
        import codecs
        with codecs.open(fn, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        # 头部按,切割
        headers = lines[0].split(',')
        # print("headers: ", headers)
        # for obj in headers:
        #     print("-----", obj)
        # 按照,分割
        lines = np.array([line.split(',') for line in lines[1:]])

        feature_size = len(headers) - 4
        genders = []
        ages = []
        values = []
        for line in lines:
            # gender = self.process_gender(line[1], aux=False)
            gender = line[1]
            # age = self.process_age(int(line[2]) if line[2] else -1, self.age_map, aux=False)
            age = int(line[2]) if line[2] else 0
            value = []
            for i in range(feature_size):
                # 从天门冬氨酸开始取
                v = line[4 + i]
                if v:
                    value.append(float(v))
                else:
                    value.append(-1)
            # if value[-1] >= 30:  # remove outlier
            #     continue
            genders.append(gender)
            ages.append(age)
            values.append(value)
            # print("value----", len(value))

        return np.array(genders), np.array(ages, dtype=np.int32), np.array(values, dtype=np.float32)


def main(_):
    # import tensorflow as tf
    # hello = tf.constant('Hello, TensorFlow!')
    # sess = tf.Session()
    # print(sess.run(hello))
    # # 'Hello, TensorFlow!'
    # a = tf.constant(10)
    # b = tf.constant(32)
    # print(sess.run(a + b))
    # # 42
    # sess.close()
    # 跑10次取平均值
    import pandas as pd
    flag = None
    for i in range(10):
        print("------- " + str(i) + " iteration begin----------")
        model = DiabetesPredictModel()
        with tf.Session() as sess:
            model.train(sess, epochs=FLAGS.epoch)

            if FLAGS.validate:
                # 计算验证集上的误差，分别为自己的loss, 采用平均值进行预测误差，提升的百分比
                loss, loss_r = model.validate(sess, print_predict=True)
                print('validate_loss=%.6f / %.6f / %.6f, validate_loss_r=%.6f' %
                      (loss, model.mean_error, loss / model.mean_error, loss_r)
                )
            # 对训练集做预测
            predict = model.test(sess, i)
            # print(len(predict))
            # print(predict)
        if not flag:
            df = predict
            flag = 1
        else:
            df = pd.concat([df, predict], axis=1)  # 按列合并
    # print(df.shape)
    df.to_csv("2018_01_26_10_model_dep8_round300.csv", header=None, index=False, float_format="%.3f")
    df_mean = df.mean(axis=1)   # 按行求均值
    df_mean.to_csv("2018_01_26_dep8_round300.csv", header=None, index=False, float_format='%.3f')


if __name__ == '__main__':
    # main("a")
    # read_data("d_train_20180102.csv")
    # argparse是python内置的一个解析模块，程序中定义好之后会自动生成帮助和使用信息
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='./mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--epoch', type=int,
                        default=600,
                        help='grid precision')  # 轮数
    parser.add_argument('--batch', type=int,
                        default=32,
                        help='batch size')   # batch的大小设置为32
    parser.add_argument('--hidden', type=int,
                        default=64,
                        help='hidden units')
    parser.add_argument('--depth', type=int,
                        default=8,
                        help='grid precision')  # 网络的层数设置为10
    parser.add_argument('--validate', type=float,
                        default=0.1,
                        help='validate portion')
    parser.add_argument('--threshold', type=float,
                        # default=0.75,
                        default=0.8,
                        help='train exit threshold')
    parser.add_argument('--activate', type=str,
                        default='tanh',
                        help='activate function')
    parser.add_argument('--optimizer', type=str,
                        default='rmsp',
                        help='optimizer')
    parser.add_argument('--skip', type=str,
                        default='true',
                        help='skip connections')
    parser.add_argument('--dropout', type=float,
                        # default=0.8,
                        default=0.85,
                        help='dropout keep rate')
    parser.add_argument('--save', type=str,
                        default='result.csv.sa',
                        help='save file')
    FLAGS, unparsed = parser.parse_known_args()
    print('depth is %s' % FLAGS.depth)
    print('hidden unit is %s' % FLAGS.hidden)
    print('batch size is %s' % FLAGS.batch)
    print('validate is %s' % FLAGS.validate)
    print('threshold is %s' % FLAGS.threshold)
    print('optimizer is %s' % FLAGS.optimizer)
    print('skip is %s' % FLAGS.skip)
    print('dropout is %s' % FLAGS.dropout)
    print('activate is %s' % FLAGS.activate)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
