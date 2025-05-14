from tensorflow.keras import layers, models, backend as K
import tensorflow as tf
import numpy as np

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            inputs, mask = inputs
            # Reshape mask to match the dimensions needed for masking
            mask = K.expand_dims(mask, -1)  # Add dimension for capsule vector
        else:
            # Get capsule lengths
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # Create one-hot mask from the longest capsule
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.shape[1])
            mask = K.expand_dims(mask, -1)
        
        # Apply mask to inputs - note inputs shape should be [batch_size, num_capsules, dim_capsule]
        masked = inputs * mask
        return K.batch_flatten(masked)  # Flatten to [batch_size, num_capsules * dim_capsule]

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        
        # Weight matrix for each output capsule
        self.W = self.add_weight(
            shape=[1, self.input_num_capsule, self.num_capsule, self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            name='W'
        )
        self.built = True

    def call(self, inputs, training=None):
        # inputs shape: [batch_size, input_num_capsule, input_dim_capsule]
        batch_size = tf.shape(inputs)[0]
        
        # Reshape inputs: [batch_size, input_num_capsule, 1, 1, input_dim_capsule]
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, 2), 2)
        
        # Tile inputs for all output capsules: 
        # [batch_size, input_num_capsule, num_capsule, 1, input_dim_capsule]
        inputs_tiled = tf.tile(inputs_expanded, [1, 1, self.num_capsule, 1, 1])
        
        # Tile weights for each batch: 
        # [batch_size, input_num_capsule, num_capsule, dim_capsule, input_dim_capsule]
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        
        # Compute predictions using batched matmul:
        # [batch_size, input_num_capsule, num_capsule, dim_capsule]
        inputs_hat = tf.matmul(W_tiled, tf.transpose(inputs_tiled, [0, 1, 2, 4, 3]))
        inputs_hat = tf.squeeze(inputs_hat, axis=-1)
        
        # Initialize routing logits
        b = tf.zeros([batch_size, self.input_num_capsule, self.num_capsule])
        
        # Routing algorithm
        for i in range(self.routings):
            # Calculate coupling coefficients (c)
            c = tf.nn.softmax(b, axis=2)
            c = tf.expand_dims(c, -1)
            
            # Weighted sum to get output capsules
            # [batch_size, num_capsule, dim_capsule]
            outputs = tf.reduce_sum(
                tf.multiply(c, inputs_hat), axis=1
            )
            outputs = squash(outputs)
            
            if i < self.routings - 1:
                # Update agreement
                outputs_tiled = tf.expand_dims(outputs, 1)
                outputs_tiled = tf.tile(outputs_tiled, [1, self.input_num_capsule, 1, 1])
                
                agreement = tf.reduce_sum(
                    tf.multiply(inputs_hat, outputs_tiled), axis=-1
                )
                b = b + agreement
        
        return outputs

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(
        filters=dim_capsule*n_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding
    )(inputs)
    
    # Calculate dimensions after conv
    output_shape = output.shape
    h, w = output_shape[1], output_shape[2]
    
    # Reshape to get capsules
    outputs = layers.Reshape(
        target_shape=[h * w * n_channels, dim_capsule]
    )(output)
    
    return layers.Lambda(squash)(outputs)

def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    
    # Conv1
    conv1 = layers.Conv2D(
        filters=256,
        kernel_size=9,
        strides=1,
        padding='valid',
        activation='relu',
        name='conv1'
    )(x)
    
    # PrimaryCaps
    primarycaps = PrimaryCap(
        conv1,
        dim_capsule=8,
        n_channels=32,
        kernel_size=9,
        strides=2,
        padding='valid'
    )
    
    # DigitCaps
    digitcaps = CapsuleLayer(
        num_capsule=n_class,
        dim_capsule=16,
        routings=routings,
        name='digitcaps'
    )(primarycaps)
    
    # Length layer for class prediction
    out_caps = Length(name='capsnet')(digitcaps)
    
    # For training model
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])
    
    # For prediction model
    masked = Mask()(digitcaps)
    
    # Decoder
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    
    # Models
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    
    return train_model, eval_model

def margin_loss(y_true, y_pred):
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5
    
    L = y_true * K.square(K.maximum(0., m_plus - y_pred)) + \
        lambda_ * (1 - y_true) * K.square(K.maximum(0., y_pred - m_minus))
    
    return K.mean(K.sum(L, 1))