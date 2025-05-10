import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4915)]
        )
    except RuntimeError as e:
        print(f"GPU 메모리 제한 설정 실패: {e}")

from tensorflow.keras import layers, Model

class ResAttnBlock(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = layers.Conv1D(dim, 3, padding='same')
        self.norm1 = layers.LayerNormalization()
        self.conv2 = layers.Conv1D(dim, 3, padding='same')
        self.norm2 = layers.LayerNormalization()
        self.attn  = layers.MultiHeadAttention(num_heads=4, key_dim=dim)

    def call(self, x):
        residual = x
        out = tf.nn.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        attn_out = self.attn(out, out)
        return tf.nn.relu(residual + attn_out)

class VoiceConversionModelTF(Model):
    def __init__(self, n_mels=80, hidden_dim=256, spk_dim=128, num_blocks=4):
        super().__init__()

        n_feats = n_mels + 1

        self.content_conv   = layers.Conv1D(hidden_dim, 5,
                                            padding='same',
                                            activation='relu',
                                            input_shape=(None, n_feats))
        self.content_blocks = [ResAttnBlock(hidden_dim) for _ in range(num_blocks)]

        self.spk_conv = [
            layers.Conv1D(256, 5, padding='same', activation='relu')
            for _ in range(2)
        ]
        self.spk_pool = layers.GlobalAveragePooling1D()
        self.spk_fc   = layers.Dense(spk_dim)

        self.adain_fc   = layers.Dense(hidden_dim * 2)
        self.dec_blocks = [ResAttnBlock(hidden_dim) for _ in range(num_blocks)]
        self.upsample   = layers.Conv1DTranspose(hidden_dim, 4,
                                                 strides=2,
                                                 padding='same',
                                                 activation='relu')
        self.out_conv   = layers.Conv1D(n_mels, 3,
                                        padding='same',
                                        activation='tanh')

    def call(self, inputs, training=False, mask=None):
        src_feat, tgt_mel = inputs
        x = self.content_conv(src_feat)
        for blk in self.content_blocks:
            x = blk(x)

        s = tgt_mel
        for conv in self.spk_conv:
            s = conv(s)
        s = self.spk_pool(s)
        s = self.spk_fc(s)

        # 3) AdaIN
        gamma_beta = self.adain_fc(s)
        gamma, beta = tf.split(gamma_beta, 2, axis=-1)
        gamma = tf.expand_dims(gamma, 1)
        beta  = tf.expand_dims(beta, 1)

        x = gamma * x + beta

        for blk in self.dec_blocks:
            x = blk(x)

        x = self.upsample(x)
        return self.out_conv(x)