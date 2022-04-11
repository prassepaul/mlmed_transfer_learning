import tensorflow as tf
from tensorflow import keras


class ContextualAttentionLayer(keras.layers.Layer):
  def __init__(self, attention_size, hidden_size, num_genes, num_gene_features=1):
    super(ContextualAttentionLayer, self).__init__()

    self.w_num_gene_features = tf.Variable(tf.random.normal([num_gene_features], stddev=0.1))
    self.w_genes = tf.Variable(tf.random.normal([num_genes, attention_size], stddev=0.1))
    self.b_genes = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
    self.w_smiles = tf.Variable(tf.random.normal([hidden_size, attention_size], stddev=0.1))
    self.b_smiles = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
    self.v = tf.Variable(tf.random.normal([attention_size], stddev=0.1))

    self.tensordotlayer1 = TensorDotLayer(axis=[2,0])
    self.tensordotlayer2 = TensorDotLayer(axis=1)
    self.reducesum = ReduceSumLayer(axis=1)
    self.softmax = keras.layers.Softmax()

    self.expanddim1 = ExpandDimLayer(axis=2)
    self.expanddim2 = ExpandDimLayer(axis=1)
    self.expanddim3 = ExpandDimLayer(axis=-1)
  
  def call(self, inputs):
    genes = self.expanddim1(inputs[0]) if len(inputs[0].shape) == 2 else inputs[0]
    genes_collapsed = self.tensordotlayer1([genes, self.w_num_gene_features])

    x = self.tensordotlayer2([genes_collapsed, self.w_genes])
    x = x + self.b_genes
    x = self.expanddim2(x)

    y = self.tensordotlayer2([inputs[1], self.w_smiles])
    y = y + self.b_smiles

    x = x + y
    x = keras.activations.tanh(x)

    xv = self.tensordotlayer2([x, self.v])
    alphas = self.softmax(xv)

    out = self.expanddim3(alphas)
    out = inputs[1] * out

    return self.reducesum(out)
    
    
class DenseAttentionLayer(keras.layers.Layer):
    def __init__(self, feature_size):
        super(DenseAttentionLayer, self).__init__()
        self.dense = keras.layers.Dense(feature_size, activation=keras.activations.softmax)

    def call(self, inputs):
        alphas = self.dense(inputs)
        return tf.multiply(inputs, alphas)
        
        
class TensorDotLayer(keras.layers.Layer):
  def __init__(self, axis):
    super(TensorDotLayer, self).__init__()
    self.axis = axis
  
  def call(self, inputs):
    return tf.tensordot(inputs[0], inputs[1], axes=self.axis)
    

class ReduceSumLayer(keras.layers.Layer):
  def __init__(self, axis):
    super(ReduceSumLayer, self).__init__()
    self.axis = axis
  
  def call(self, input):
    return tf.reduce_sum(input, axis=self.axis)
    
    
class ExpandDimLayer(keras.layers.Layer):
  def __init__(self, axis):
    super(ExpandDimLayer, self).__init__()
    self.axis = axis
  
  def call(self, input):
    return keras.backend.expand_dims(input, axis=self.axis)
    
class SqueezeLayer(keras.layers.Layer):
  def __init__(self, axis):
    super(SqueezeLayer, self).__init__()
    self.axis = axis
  
  def call(self, input):
    return keras.backend.squeeze(input, axis=self.axis)
    
class EmbeddingLayer(keras.layers.Layer):
  def __init__(self, x, y):
    super(EmbeddingLayer, self).__init__()
    
    self.embedding_matrix = tf.Variable(tf.random.normal((x, y)))
  
  def call(self, input):
    return tf.nn.embedding_lookup(self.embedding_matrix, tf.cast(input, dtype=tf.int32))
    
def get_paccmann_model(params):
  input_smiles = keras.Input(shape=(params["smiles_length"],), name="input_smiles")
  embedding_smiles = EmbeddingLayer(params["smiles_vocab"], params["smiles_embedding_size"]) (input_smiles)
  smiles_expand = ExpandDimLayer(axis=3) (embedding_smiles)

  input_zeros = keras.Input(shape=(1,), name="input_zeros")
  pad = EmbeddingLayer(params["smiles_vocab"], params["smiles_embedding_size"]) (input_zeros)
  pad = ExpandDimLayer(axis=3) (pad)

  convolved_smiles = []
  for index, (filter_size, kernel_size) in enumerate(zip(params["filter"], params["kernels"])):
    smiles_pad = keras.layers.concatenate([pad]*(kernel_size[0] // 2) + [smiles_expand] + [pad]*(kernel_size[0] // 2), axis=1)

    conv_smiles = keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size, activation=tf.nn.relu) (smiles_pad)
    conv_smiles = SqueezeLayer(axis=2) (conv_smiles)
    conv_smiles = keras.layers.Dropout(rate=params["dropout"]) (conv_smiles)
    convolved_smiles.append(keras.layers.BatchNormalization() (conv_smiles)) 

  convolved_smiles.insert(0, embedding_smiles)

  input_genes = keras.Input(shape=(params["genes_number"],), name="input_genes")
  encoded_genes = [DenseAttentionLayer(params["genes_number"])(input_genes) for i in range(len(params["multiheads"]))]

  encoding_coefficients = [ContextualAttentionLayer(
                              attention_size=params["smiles_attention_size"], 
                              hidden_size=convolved_smiles[layer].shape[2],
                              num_genes=params["genes_number"]) ([encoded_genes[layer], convolved_smiles[layer]])
                          for layer in range(len(convolved_smiles)) for _ in range(params["multiheads"][layer])]

  encoding = keras.layers.concatenate(encoding_coefficients, axis=1)
  encoding = keras.layers.Reshape((params["smiles_embedding_size"] * params["multiheads"][0] + sum([a*b for a,b in zip(params["multiheads"][1:], params["filter"])]),)) (encoding)

  x = keras.layers.BatchNormalization() (encoding) 

  for index, size in enumerate(params["stacked_dense_hidden_sizes"]):
    x = keras.layers.Dense(size, activation=None) (x)
    x = keras.layers.BatchNormalization() (x) 
    x = keras.layers.ReLU() (x)
    x = keras.layers.Dropout(rate=params["dropout"]) (x) 

  output = keras.layers.Dense(1) (x)

  model = keras.Model(inputs=[input_smiles, input_zeros, input_genes], outputs=output)
  opt = tf.keras.optimizers.Adam()#learning_rate=learning_rate)
  loss = tf.keras.losses.MeanSquaredError()
  model.compile(optimizer = opt, loss = loss, 
                metrics=["mse"])
  return model
    