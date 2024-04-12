import tensorflow as tf
from keras import layers, Model
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.models import Sequential
from HyNN.layers import Encoder

class PatchEmbeddings(layers.Layer):
    """
    Patch embeddings layer
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        """
        * `d_model` is the transformer embeddings size
        * `patch_size` is the size of the patch
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch.
        self.conv = tf.keras.layers.Conv2D(
            filters=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            input_shape=(None, None, in_channels)
        )


    def __call__(self, x):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Apply the convolution layer to get the transformed "patches" [batch_size, h, w, c]
        x = self.conv(x)

        # Rearrange to shape `[batch_size, patches, d_model]` [n, h*w, c]
        n, h, w, c = x.shape
        x = tf.reshape(x, [-1, h * w, c])

        return x


class LearnedPositionalEmbeddings(layers.Layer):
    """
    <a id="LearnedPositionalEmbeddings"></a>

    ## Add parameterized positional encodings

    This adds learned positional embeddings to patch embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5_000):
        """
        * `d_model` is the transformer embeddings size
        * `max_len` is the maximum number of patches
        """
        super().__init__()
        # Positional embeddings for each location
        self.positional_encodings = tf.Variable(tf.zeros((1, max_len, d_model)), trainable=True)

    def __call__(self, x):
        """
        * `x` is the patch embeddings of shape `[patches, batch_size, d_model]`
        """
        # Get the positional embeddings for the given patches
        pe = self.positional_encodings[:,:x.shape[1],:]
        # Add to patch embeddings and return
        return x + pe



class VisionTransformer(Model):
    """
    ## Vision Transformer

    This combines the [patch embeddings](#PatchEmbeddings),
    [positional embeddings](#LearnedPositionalEmbeddings),
    transformer and the [classification head](#ClassificationHead).
    """
    def __init__(self, d_model: int, patch_size: int, in_channels: int, n_heads: int,n_layers: int, bias: bool = False, dropout: float = 0.1):
        """
        * `transformer_layer` is a copy of a single [transformer layer](../models.html#TransformerLayer).
         We make copies of it to make the transformer with `n_layers`.
        * `n_layers` is the number of [transformer layers](../models.html#TransformerLayer).
        * `patch_emb` is the [patch embeddings layer](#PatchEmbeddings).
        * `pos_emb` is the [positional embeddings layer](#LearnedPositionalEmbeddings).
        * `classification` is the [classification head](#ClassificationHead).
        """
        super().__init__()
        # Patch embeddings
        self.patch_emb = PatchEmbeddings(
            d_model,
            patch_size,
            in_channels
        
        )
        self.pos_emb = LearnedPositionalEmbeddings(d_model)

        # Transformer layers
        self.encoder = Encoder(
            attention_num_heads=n_heads,
            attention_key_dim= d_model,
            attention_value_dim= d_model,
            attention_output_dim= d_model,
            attention_dropout= dropout,
            ffn_hidden_size=d_model*4,
            num_layers=n_layers,
            attention_use_bias=bias,
        )

        # `[CLS]` token embedding
        self.cls_token_emb = tf.Variable(tf.random.uniform([1, 1, d_model]), trainable=True)
        # Final normalization layer
        self.ln = tf.keras.layers.LayerNormalization()

        self.mlp = Sequential([
            Dense(128, input_shape=(d_model,)),  # Start with your input dimension
            BatchNormalization(),  # Apply batch normalization
            Activation('relu'),  # Then apply the activation function
            Dropout(dropout),  # Apply dropout with a rate of 0.5 (adjust as necessary)
            
            Dense(64),
            BatchNormalization(),
            Activation('relu'),
            Dropout(dropout),  # Adjust dropout rate as necessary
            
            Dense(32),
            BatchNormalization(),
            Activation('relu'),
            Dropout(dropout),  # Adjust dropout rate as necessary
            
            Dense(16, activation='relu')  # Final layer with activation
        ])

    def __call__(self, x):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Get patch embeddings. This gives a tensor of shape `[patches, batch_size, d_model]`
        x = self.patch_emb(x)
        #print("patch_emb shape: ", x.shape)
        # Concatenate the `[CLS]` token embeddings before feeding the transformer
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.tile(self.cls_token_emb, [batch_size, 1, 1])
        x = tf.concat([cls_tokens, x], axis=1)
        #print("concat shape: ", x.shape)
        # Add positional embeddings
        x = self.pos_emb(x)
        #print("pos_emb shape: ", x.shape)
        # Pass through transformer layers with no attention masking
        x = self.encoder(x)
        #print("encoder shape: ", x.shape)
        # Get the transformer output of the `[CLS]` token (which is the first in the sequence).
        x = x[:,0]
        #print("cls shape: ", x.shape)
        mlp_output = self.mlp(x)
        #print("mlp_output shape: ", mlp_output.shape)
        return mlp_output