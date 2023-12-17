import tensorflow as tf
#DenseBlock Layer
class DenseBlock(tf.keras.Model):
    def __init__(self, bottleneck_num, growth_rate):
        super(DenseBlock, self).__init__(name="")
        self.bottleneck_layers = self._build_bottlenecks(bottleneck_num, growth_rate)

    def _build_bottlenecks(self, bottleneck_num, growth_rate):
        bottleneck_layers = []
        #Based on the convolutional num to build corresponding num of bottleneck
        for _ in range(bottleneck_num):
            layers = [
                tf.keras.layers.BatchNormalization(), #Normalization layer
                tf.keras.layers.ReLU(), #ReLU layer
                tf.keras.layers.Conv2D(growth_rate * 4, (1, 1)),#the 1*1 convolution kernel remain consistent without padding
                tf.keras.layers.BatchNormalization(),#Normalization layer
                tf.keras.layers.ReLU(), #ReLu layer
                tf.keras.layers.Conv2D(growth_rate, (3, 3), padding="same")
            ]
            bottleneck_layers.append(layers)
        return bottleneck_layers

    def call(self, input, training=False):
        x = input
        output = None
        first_iteration = True

        for layers in self.bottleneck_layers:
            x = self._apply_layers(x, layers)
            #Check whether the progress is in first iteration
            if first_iteration:
                output = tf.concat([x, input], axis=-1)
                first_iteration = False
            else:
                output = tf.concat([x, output], axis=-1)

            x = output

        return output

    def _apply_layers(self, x, layers):
        for layer in layers:
            x = layer(x)
        return x


class Transition(tf.keras.Model):
    def __init__(self, out_channel):
        super(Transition, self).__init__(name="")
        self.BN = tf.keras.layers.BatchNormalization()
        self.Conv = tf.keras.layers.Conv2D(out_channel, (1, 1))
        self.Pool = tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, input, training=False):
        x = self.BN(input)
        x = self.Conv(x)
        x = self.Pool(x)
        return x
#the output : the probability for 4 classes separately
class Classification(tf.keras.Model):
    def __init__(self, class_num=4): #4 classes of lables results in 4 classes output
        super(Classification, self).__init__(name="")
        self.global_avgpool = tf.keras.layers.GlobalAvgPool2D()
        self.Dense = tf.keras.layers.Dense(class_num)
        self.Softmax = tf.keras.layers.Softmax()

    def call(self, input, training=False):
        x = self.global_avgpool(input)
        x = self.Dense(x)
        x = self.Softmax(x)
        return x
#initialize the whole model
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__(name="DenseNetModel")
        self.Conv = tf.keras.layers.Conv2D(16, (6, 6), strides=(3, 3)) #[1,64,64,16]
        self.Pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)) #[1,32,32,16]
        self.DB1 = DenseBlock(8, 24)
        self.Transition1 = Transition((64 + 192) // 2)
        self.DB2 = DenseBlock(8, 24)
        self.Transition2 = Transition(((64 + 192) // 2 + 192) // 2)
        self.DB3 = DenseBlock(8, 24)
        self.Transition3 = Transition((((64 + 192) // 2 + 192) // 2 + 192) // 2)
        self.Classification = Classification(4)

    def call(self, input):
        x = self.Conv(input)
        x = self.Pool(x)
        x = self.DB1(x)
        x = self.Transition1(x)
        x = self.DB2(x)
        x = self.Transition2(x)
        x = self.DB3(x)
        x = self.Transition3(x)
        x = self.Classification(x)
        return x

#record the callback to the logs file
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super(CustomCallback, self).__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.log_file, 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}: {logs}\n")
