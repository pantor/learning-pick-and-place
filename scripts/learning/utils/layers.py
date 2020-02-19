import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401
import tensorflow.keras.backend as tkb  # pylint: disable=E0401


class PartialConv2D(tkl.Conv2D):
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold
        self.strides = kwargs['strides']
        self.filters = kwargs['filters']
        self.kernel_size = kwargs['kernel_size']
        super(PartialConv2D, self).__init__(**kwargs)

    def call(self, x):
        mask = tkb.tf.to_float(x > 0)
        unit_value = 1.0 / (self.kernel_size[0] * self.kernel_size[1])
        mask_update = tkb.tf.layers.conv2d(
            mask,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            kernel_initializer=tkb.tf.constant_initializer(unit_value),
            trainable=False
        )

        x = super(PartialConv2D, self).call(x)
        return tkb.tf.where(
            mask_update > self.threshold,
            tkb.tf.multiply(x, tkb.tf.clip_by_value(mask_update, 1e-7, 1.0)),
            tkb.tf.zeros_like(mask_update)
        )


def dense_block_gen(l2_reg=0.01, dropout_rate=0.4):
    def dense_block(x, size, dropout_rate=dropout_rate):
        x = tkl.Dense(size, kernel_regularizer=tk.regularizers.l2(l2_reg))(x)
        x = tkl.LeakyReLU()(x)
        x = tkl.BatchNormalization()(x)
        return tkl.Dropout(rate=dropout_rate)(x)
    return dense_block


def conv_block_gen(l2_reg=0.01, dropout_rate=0.4, monte_carlo=None, bias_initializer=None):
    def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), l2_reg=l2_reg):
        x = tkl.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            bias_initializer=bias_initializer,
            kernel_regularizer=tk.regularizers.l2(l2_reg),
        )(x)
        x = tkl.LeakyReLU()(x)
        x = tkl.BatchNormalization()(x)
        return tkl.Dropout(rate=dropout_rate)(x, training=monte_carlo)
    return conv_block


def one_hot_gen(num_classes):
    def one_hot(x):
        return tkb.one_hot(tkb.cast(x, 'uint8'), num_classes=num_classes)
    return one_hot


def sampling(args):
    z_mean, z_logvar = args
    epsilon = tkb.random_normal(tkb.shape(z_mean))
    return z_mean + tkb.exp(0.5 * z_logvar) * epsilon
