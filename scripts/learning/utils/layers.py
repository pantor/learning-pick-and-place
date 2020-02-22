import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401
import tensorflow.keras.backend as tkb  # pylint: disable=E0401


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
