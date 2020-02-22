import tensorflow as tf
import tensorflow.keras as tk  # pylint: disable=E0401


class Split:
    confident_learning_alpha = None  # in [0, 1]

    @classmethod
    def single_class_split(cls, y_true, y_pred):
        value_true = y_true[:, 0]
        index = tf.cast(y_true[:, 1], tf.int32)
        sample_weight = y_true[:, 2]

        indices_class = tf.stack([tf.range(tf.shape(input=index)[0]), index], axis=1)
        value_pred = tf.gather_nd(y_pred, indices_class)

        return tf.expand_dims(value_true, 1), tf.expand_dims(value_pred, 1), sample_weight


class SplitCallback(tk.callbacks.Callback):
    def __init__(self, min_epoch, step_per_epoch=1e-3, verbose=1):
        self.min_epoch = min_epoch
        self.step_per_epoch = step_per_epoch
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs):
        if epoch > self.min_epoch:
            Split.confident_learning_alpha = self.step_per_epoch * (epoch - self.min_epoch)

            if self.verbose > 0:
                print(f'Set split confident learning alpha to: {Split.confident_learning_alpha}')


class Losses:
    def __init__(self):
        self.bce = tk.losses.BinaryCrossentropy()
        self.mse = tk.losses.MeanSquaredError()

        self.margin = 5.0

    def binary_crossentropy(self, y_true, y_pred):
        value_true, value_pred, sample_weight = Split.single_class_split(y_true, y_pred)
        return self.bce(value_true, value_pred, sample_weight=sample_weight)

    def mean_square_error(self, y_true, y_pred):
        value_true, value_pred, sample_weight = Split.single_class_split(y_true, y_pred)
        return self.mse(value_true, value_pred, sample_weight=sample_weight)

    def contrastive_loss(self, y_true, y_pred):
        value_true, value_pred, sample_weight = Split.single_class_split(y_true, y_pred)
        loss = value_true * value_pred + (1 - value_true) * tk.backend.maximum(0.0, self.margin - value_pred)
        return tk.backend.mean(loss * sample_weight)


class SplitBinaryCrossentropy(tk.losses.BinaryCrossentropy):
    def __init__(self, name='sbinary_crossentropy'):
        super(SplitBinaryCrossentropy, self).__init__(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        value_true, value_pred, sample_weight = Split.single_class_split(y_true, y_pred)
        return super().__call__(value_true, value_pred, sample_weight=sample_weight)


class SplitMeanSquaredError(tk.losses.MeanSquaredError):
    def __init__(self, name='smean_squared_error'):
        super(SplitMeanSquaredError, self).__init__(name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        value_true, value_pred, sample_weight = Split.single_class_split(y_true, y_pred)
        return super().__call__(value_true, value_pred, sample_weight=sample_weight)


class SplitBinaryAccuracy(tk.metrics.BinaryAccuracy):
    def __init__(self, name='sbinary_accuracy', dtype=None, threshold=0.5):
        super(SplitBinaryAccuracy, self).__init__(name=name, dtype=dtype, threshold=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        value_true, value_pred, sample_weight = Split.single_class_split(y_true, y_pred)
        super().update_state(value_true, value_pred, sample_weight=sample_weight)


class SplitPrecision(tk.metrics.Precision):
    def __init__(self, thresholds=None, top_k=None, class_id=None, name='sprecision', dtype=None):
        super(SplitPrecision, self).__init__(thresholds=thresholds, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        value_true, value_pred, sample_weight = Split.single_class_split(y_true, y_pred)
        super().update_state(value_true, value_pred, sample_weight=sample_weight)


class SplitRecall(tk.metrics.Recall):
    def __init__(self, thresholds=None, top_k=None, class_id=None, name='srecall', dtype=None):
        super(SplitRecall, self).__init__(thresholds=thresholds, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        value_true, value_pred, sample_weight = Split.single_class_split(y_true, y_pred)
        super().update_state(value_true, value_pred, sample_weight=sample_weight)
