# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A function to build and compile a Transformer model.

This module provides a primary function to assemble the model layers,
configure the optimizer and loss, and return a compiled Keras model ready for
training.
"""

from typing import Literal, Any, Callable, Optional, List, Dict

import keras
from keras import layers, ops
import jax
from jax import numpy as jnp
import random 
import numpy as np


def create_model(
    vocabulary_size: int,
    max_length: int,
    embedding_dim: int = 256,
    mlp_dim: int = 256,
    num_heads: int = 2,
    num_blocks: int = 1,
    optimizer: Literal["adamw", "sgd"] = "adamw",
    learning_rate: float = 5e-4,
    dropout_rate: float = 0.0,
    activation_function: str = "relu",
    pad_token_id: int = 0,
) -> keras.Model:
  """Creates a transformer-based model for sequence processing tasks.

  Example:
    model = create_model(vocabulary_size=5000, max_length=100,
                         embedding_dim=256, mlp_dim=512,
                         num_heads=8, num_blocks=2)
    print(model.summary())

  Notes:
    - The model uses causal (masked) attention to ensure that each token only
      attends to previous tokens and not future tokens.
    - The final dense layer produces a logit over the vocabulary for each token
      in the sequence.
    - The loss function is `CustomMaskPadLoss`, which ignores padding tokens in
      the loss computation.

  Args:
    vocabulary_size: The size of the vocabulary, i.e., the number of unique
      tokens.
    max_length: The maximum length of the input sequences.
    embedding_dim: The dimensionality of the embedding space.
    mlp_dim: The number of units in the feed-forward network of each transformer
      block.
    num_heads: The number of attention heads in the multi-head attention
      mechanism.
    num_blocks: The number of transformer blocks to stack in the model.
    optimizer: The optimizer to use for training, either 'adamw' (Adam with
      weight decay) or 'sgd'.
    learning_rate: The learning rate for the optimizer.
    dropout_rate: The dropout rate to prevent overfitting.
    activation_function: The activation function to use in the feed-forward
      network of each transformer block.
    pad_token_id: The ID used to represent padding tokens in the sequence. This
      is used to mask padded tokens in the loss calculation.

  Returns:
    The compiled Keras model which outputs the probability of the next token
        prediction.

  Raises:
      NotImplementedError: If an unsupported optimizer is specified.
  """
  # Create input layer.
  inputs = layers.Input(shape=(max_length,), dtype="int32")

  # Embedding layer that combines token and positional embeddings.
  embedding_layer = TokenAndPositionEmbedding(
      max_length, vocabulary_size, embedding_dim
  )
  x = embedding_layer(inputs)

  # Apply a stack of transformer blocks.
  for _ in range(num_blocks):
    transformer_block = TransformerBlock(
        embedding_dim,
        num_heads,
        mlp_dim,
        dropout_rate=dropout_rate,
        activation_function=activation_function,
    )
    x = transformer_block(x)

  # Apply dense layer, it returns raw logit of next token prediction.
  outputs = layers.Dense(vocabulary_size)(x)

  # Build the model.
  model = keras.Model(inputs=inputs, outputs=outputs)

  # Set up optimizer based on input string.
  optimizer_instance = get_optimizer(optimizer, learning_rate)

  # Define the loss function and compile the model.
  loss_fn = CustomMaskPadLoss(pad_token_id=pad_token_id)
  model.compile(optimizer=optimizer_instance, loss=loss_fn)

  # Final output layer returns the probability of next token prediction.
  return model


def get_optimizer(
    optimizer_name: Literal["adamw", "sgd"], learning_rate: float
) -> keras.optimizers.Optimizer:
  """Helper function to get the appropriate optimizer instance.

  Args:
    optimizer_name: The name of the optimizer.
    learning_rate: The learning rate for the optimizer.

  Returns:
    The corresponding optimizer instance.

  Raises:
    NotImplementedError: If an unsupported optimizer is specified.
  """

  if optimizer_name.lower() == "sgd":
    return keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer_name.lower() == "adamw":
    return keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.005,
        gradient_accumulation_steps=None,
    )
  else:
    raise NotImplementedError(f"Optimizer {optimizer_name} is not implemented.")

# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class CustomMaskPadLoss(keras.losses.Loss):
  """Custom loss function for masked padding in sequence-based tasks.

  This loss function computes the SparseCategoricalCrossentropy
  loss while ignoring the padding tokens (specified by `pad_token_id`).
  The padding tokens are not included in the loss calculation,
  allowing the model to focus on meaningful tokens during training.

  Attributes:
    name: The name of the loss function, used by Keras.
    pad_token_id: The ID of the padding token. If provided, padding tokens will
      be ignored during loss calculation. If None, no padding is masked.
    **kwargs: Additional keyword arguments.
  """

  def __init__(
      self,
      pad_token_id: int | None = None,
      **kwargs: Any,
  ):
    super().__init__(name="custom_mask_pad_loss", **kwargs)
    self.pad_token_id = pad_token_id

  def call(self, y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Computes the custom loss.

    The call function optionally masks the padding tokens and normalizes
    the loss by the number of non-masked tokens. The loss is computed using
    the SparseCategoricalCrossentropy loss function.

    Args:
      y_true: The true labels.
      y_pred: The model's predictions.

    Returns:
      The computed loss.
    """
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        # The model's output is a probability distribution. If
        # it is raw logit, this should be True.
        from_logits=True,
        ignore_class=self.pad_token_id,
        # Average the loss across the batch size.
        reduction="sum_over_batch_size",
    )

    loss = loss_fn(y_true, y_pred)
    return loss

# This value is a standard choice, as set in the
# "Attention Is All You Need" paper.
ANGLE_RATE_MULTIPLIER = 10000


# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
  """Combines token embeddings with positional embeddings.

  This layer creates combined token and positional embeddings for input
  sequences. The `mask_zero=True` setting in the token embeddings allows for
  automatic masking of padded tokens.

  Attributes:
    max_length: The maximum expected sequence length. This determines the range
      of positional embeddings.
    vocabulary_size: The size of the vocabulary. This determines the size of the
      token embedding matrix.
    embedding_dim: The dimensionality of the token and positional embeddings.
    positional_embedding_type: The type of positional embedding to use. It can
      be 'simple' or 'sinusoidal'.

  Call Arguments:
    x: Input tensor of shape (batch_size, sequence_length).

  Returns:
    jax.Array: Output tensor of shape (batch_size, sequence_length, d_model)
        with token and positional embeddings combined.
  """

  def __init__(
      self,
      max_length: int,
      vocabulary_size: int,
      embedding_dim: int,
      positional_embedding_type: str = "sinusoidal",
  ):
    super().__init__()

    self.embedding_dim = embedding_dim
    self.max_length = max_length
    self.positional_embedding_type = positional_embedding_type

    # Set mask_zero=True so that Keras generates a mask for padded tokens.
    self.token_emb = layers.Embedding(
        input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True
    )

    if self.positional_embedding_type == "simple":
      self.pos_emb = layers.Embedding(
          input_dim=max_length, output_dim=embedding_dim
      )
    elif self.positional_embedding_type == "sinusoidal":
      self.pos_emb = self.positional_encoding(
          length=max_length, depth=embedding_dim
      )
    else:
      raise NotImplementedError(
          "Positional embedding type"
          f" {self.positional_embedding_type}"
          " not implemented."
      )

  def positional_encoding(
      self, length: int, depth: int
  ) -> Callable[[Any], jax.Array]:
    """Creates a positional encoding for a sequence of tokens.

    This approach uses sine and cosine functions at varying frequencies to
    create a unique positional representation for each token in the sequence.

    Args:
      length: The length of the sequence (number of tokens).
      depth: The dimensionality of the encoding (must be even).

    Returns:
      A function that returns an array of shape (length, depth) representing
      the positional encoding. This is a function to make it compatible with the
      simple embedding layer.
    """

    depth = depth // 2  # Use integer division to ensure an integer depth.

    positions = jnp.arange(length)[:, jnp.newaxis]  # (seq, 1)
    depths = jnp.arange(depth)[jnp.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (ANGLE_RATE_MULTIPLIER**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = jnp.concatenate(
        [jnp.sin(angle_rads), jnp.cos(angle_rads)], axis=-1
    )

    pos_encoding_matrix = ops.cast(pos_encoding, dtype="float32")

    def apply(*args) -> jax.Array:  # pylint: disable=unused-argument
      return pos_encoding_matrix[jnp.newaxis, :, :]

    return apply

  def call(self, x: jax.Array) -> jax.Array:
    """Applied and combines token embeddings with positional embeddings.

    Args:
      x: Input tensor of shape (batch_size, sequence_length).

    Returns:
      Output tensor of shape (batch_size, sequence_length, d_model) with token
          and positional embeddings combined.
    """
    token_embeddings = self.token_emb(x)

    if self.positional_embedding_type == "sinusoidal":
      # This factor sets the relative scale of the embedding
      # and positonal_encoding.
      token_embeddings *= ops.sqrt(
          ops.cast(self.embedding_dim, dtype="float32")
      )
      position_embeddings = self.pos_emb(None)
    else:
      # Defaults to simple `positional_embedding_type`.
      positions = ops.arange(0, self.max_length, 1)
      position_embeddings = self.pos_emb(positions)

    return token_embeddings + position_embeddings

# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
  """A single transformer block.

  The transformer block is a fundamental component of the transformer
  architecture, which is commonly used for sequence-based tasks. It consists
  of a MultiHeadAttention layer followed by a feed-forward network,
  with layer normalization and dropout applied at each step.

  Example:
    transformer_block = TransformerBlock(embedding_dim=256, num_heads=8,
                                         mlp_dim=1024)
    output = transformer_block(inputs)

  Attributes:
    embedding_dim: The dimensionality of the input embedding (also the output
      size of the attention layer).
    num_heads: The number of attention heads in the multi-head attention
      mechanism.
    mlp_dim: The number of units in the feed-forward network.
    dropout_rate: Dropout rate, between 0 and 1.
    activation_function: The activation function to use in the feed-forward
      network.
    seed: Random seed for dropout and attention layers to ensure
      reproducibility.

  Call Arguments:
    inputs: Input tensor of shape (batch_size, sequence_length, d_model).

  Returns:
    The output of the Transformer block after applying the multi-head attention,
        feed-forward network, layer normalization, and residual connections.
  """

  def __init__(
      self,
      embedding_dim: int,
      num_heads: int,
      mlp_dim: int,
      dropout_rate: float = 0.0,
      activation_function: str = "relu",
  ):
    super().__init__()

    self.self_attention = MultiHeadSelfAttention(
        embedding_dim, num_heads, dropout_rate
    )
    self.feed_forward = FeedForwardNetwork(
        embedding_dim, mlp_dim, dropout_rate, activation_function
    )

  def call(self, inputs: jax.Array) -> jax.Array:
    """Applies a single transformer block to the input tensor.

    Notes:
      - The transformer block follows the architecture with residual connections
        and layer normalization.

    Args:
      inputs: The input tensor of shape (batch_size, seq_len, embed_dim).

    Returns:
      The output tensor of shape (batch_size, seq_len, embed_dim) after applying
          the transformer block.
    """

    # First block: masked self-attention.
    attn_output = self.self_attention(inputs)

    # Second block: feedforward network applied on attention output.
    ffn_output = self.feed_forward(attn_output)

    return ffn_output

# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class FeedForwardNetwork(layers.Layer):
  """Feed forward network layer.

  This layer implements a two-layer feedforward network with a residual
  connection and layer normalization. It's a common component in transformer
  architectures, used to introduce non-linearity and improve the model's ability
  to capture complex relationships.

  Attributes:
    embedding_dim: The dimensionality of the embedding space.
    mlp_dim: The dimensionality of the hidden layer in the feedforward network
      (often larger than embedding_dim).
    dropout_rate: The dropout rate applied to the output of the feedforward
      network.
    activation_function: The activation function used in the first dense layer.

  Call Arguments:
    x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

  Returns:
    Output tensor of shape (batch_size, sequence_length, embedding_dim) with
        the feed-forward network applied.
  """

  def __init__(
      self,
      embedding_dim: int,
      mlp_dim: int,
      dropout_rate: float = 0.0,
      activation: str = "relu",
  ):
    super().__init__()
    # Define a two-layer feedforward network.
    self.ffn = keras.Sequential([
        # Expand dimension.
        layers.Dense(mlp_dim, activation=activation),
        # Project back to embedding_dim.
        layers.Dense(embedding_dim),
    ])
    self.dropout = layers.Dropout(dropout_rate)
    self.layernorm = layers.LayerNormalization()

  def call(self, x: jax.Array) -> jax.Array:
    """Applies the feedforward network to the input tensor.

    Args:
      x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

    Returns:
      Output tensor of shape (batch_size, sequence_length, embedding_dim).
    """

    ffn_output = self.ffn(x)
    ffn_output = self.dropout(ffn_output)
    # Add residual connection followed by layer normalization.
    output = self.layernorm(x + ffn_output)
    return output  # type: ignore


# Decorator so that the custom class can be saved and loaded correctly.
@keras.saving.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
  """Multi-head self-attention Layer.

  This layer implements multi-head self-attention, a key component in
  transformer architectures. It computes attention weights for each head and
  applies them to the input to generate a contextually enriched representation.

  Attributes:
    embedding_dim: The dimensionality of the embedding space.
    num_heads: The number of attention heads.
    dropout_rate: The dropout rate applied to the attention output.

  Call Arguments:
    x: Input tensor of shape (batch_size, sequence_length, d_model).

  Returns:
    Output tensor of shape (batch_size, sequence_length, embedding_dim)
        with self-attention applied.
  """

  def __init__(
      self, embedding_dim: int, num_heads: int, dropout_rate: float = 0.0
  ):
    super().__init__()

    # Multi-head self-attention layer.
    self.mha = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embedding_dim
    )
    self.dropout = layers.Dropout(dropout_rate)
    self.layernorm = layers.LayerNormalization()

  def call(self, x: jax.Array) -> jax.Array:
    """Applies multi-head self-attention to the input tensor.

    Args:
      x: Input tensor of shape (batch_size, sequence_length, embedding_dim).

    Returns:
      Output tensor of shape (batch_size, sequence_length, embedding_dim).
    """

    # Apply self-attention. The mask is typically a look-ahead mask.
    attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
    attn_output = self.dropout(attn_output)

    # Add residual connection followed by layer normalization.
    output = self.layernorm(x + attn_output)

    return output

def sampling(probs: jax.Array, key: jax.Array) -> int:
  """Sample a token index from the predicted next token probability.

  Args:
    probs: The probability distribution of predicted next token.
    key: The JAX random key.

  Returns:
    The index of the sampled token.
  """
  probs_array = np.array(probs)
  token_index = jax.random.choice(key, jnp.arange(probs.shape[0]), p=probs_array)
  return int(token_index)


def greedy_decoding(probs: jax.Array) -> int:
  """Select the token index from the predicted next token probability.

  Args:
    probs: The probability distribution of predicted next token.

  Returns:
    The index of the token with the highest probability.
  """
  probs_array = np.array(probs)
  return int(jnp.argmax(probs_array))


def generate_text(
    start_prompt: str,
    n_tokens: int,
    model: keras.Model,
    tokenizer: Any,
    pad_token_id: int = 0,
    sampling_mode: Literal["random", "greedy"] = "random",
) -> tuple[str, list[jax.Array]]:
  """Generate text based on a starting prompt using a trained model.

  Args:
    start_prompt: The initial prompt to start the generation.
    n_tokens: The number of tokens to generate after the prompt.
    model: The trained model to use for text generation.
    tokenizer: The tokenizer to encode and decode text.
    pad_token_id: The token ID used for padding.
    sampling_mode: Whether to use random or greedy sampling. Supported options
      are 'random' and 'greedy'.

  Returns:
    The generated text after the prompt.
  """

  if sampling_mode not in ["random", "greedy"]:
    raise ValueError(
        f"Sampling mode {sampling_mode} is not supported. Supported options are"
        " 'random' and 'greedy'."
    )

  # Introduce randomness by re-intializing JAX RNG with a different seed on
  # each call. While this harms reproducability, it avoids having to pass a JAX
  # key on every call, which would likely be confusing to learners.
  main_key = jax.random.PRNGKey(random.randint(0, 1000000))

  max_length = model.layers[0].output.shape[1]

  # Tokenize the starting prompt.
  start_tokens = tokenizer.encode(start_prompt)

  # Generate tokens.
  tokens_generated = start_tokens + []
  probs = []
  for _ in range(n_tokens):
    pad_len = max_length - len(start_tokens)
    sample_index = len(start_tokens) - 1
    if pad_len < 0:
      # Truncate the input sequence to fit the max context length.
      x = start_tokens[:max_length]
      sample_index = max_length - 1
    elif pad_len > 0:
      x = start_tokens + [pad_token_id] * pad_len  # Pad the input sequence.
    else:
      x = start_tokens

    x = jnp.array([x])

    # Get predictions from the model.
    y = model.predict(x, verbose="0")

    # Apply softmax to convert logits to probabilities.
    probabilities = ops.softmax(y, axis=-1)

    probs.append(probabilities[0][sample_index])

    # Use greedy decoding or sampling based on sampling_mode.
    if sampling_mode == "greedy":
      sample_token = greedy_decoding(probabilities[0][sample_index])
    else:
      key, main_key = jax.random.split(main_key)
      sample_token = sampling(probabilities[0][sample_index], key)

    tokens_generated.append(sample_token)
    start_tokens.append(sample_token)

  # Convert tokens back to text.
  generated_text = tokenizer.decode(tokens_generated)
  generated_text = generated_text.replace(tokenizer.decode([pad_token_id]), "")

  return generated_text, probs


class TextGenerator(keras.callbacks.Callback):
  """A callback to generate text from a trained model.

    1. Feed an initial prompt to the model.
    2. Predict probabilities for the next token.
    3. Sample the next token and add it to the input for the next prediction.

  Attributes:
    max_tokens: Number of tokens to generate.
    start_tokens: Token indices for the initial prompt.
    tokenizer: The tokenizer used to decode generated token indices.
    pad_token_id: The padding token ID.
    print_every: Print the generated text every `print_every` epochs.
    **callback_kwargs: Any additional keyword arguments.
  """

  def __init__(
      self,
      max_tokens: int,
      start_tokens: List[int],
      tokenizer: Any,
      pad_token_id: int = 0,
      print_every: int = 1,
      **callback_kwargs: Dict[str, Any],
  ):
    super().__init__(**callback_kwargs)

    self.max_tokens = max_tokens
    self.start_tokens = start_tokens
    self.tokenizer = tokenizer
    self.print_every = print_every
    self.pad_token_id = pad_token_id  # ID for padding token.

  def on_epoch_end(
      self, epoch: int, logs: Dict[str, Any] | None = None
  ) -> None:
    """Generate and print text after each epoch based on starting tokens.

    Args:
      epoch: The current epoch number.
      logs: Logs from the training process.
    """

    if self.model is None:
      return

    max_length = self.model.layers[0].output.shape[1]
    # Make a copy of the start tokens.
    start_tokens = list(self.start_tokens)
    if (epoch + 1) % self.print_every != 0:
      return

    num_tokens_generated = 0
    tokens_generated = []

    # Introduce randomness by re-intializing JAX RNG with a different seed on
    # each call. While this harms reproducability, it avoids having to pass a
    # JAX key on every call, which would likely be confusing to learners.
    main_key = jax.random.PRNGKey(random.randint(0, 1000000))

    while num_tokens_generated < self.max_tokens:
      pad_len = max_length - len(start_tokens)
      sample_index = len(start_tokens) - 1

      # Handle padding to ensure the sequence is of the correct length.
      if pad_len < 0:
        x = start_tokens[:max_length]
        sample_index = max_length - 1
      elif pad_len > 0:
        x = start_tokens + [self.pad_token_id] * pad_len
      else:
        x = start_tokens

      x = jnp.array([x])
      y = self.model.predict(x, verbose=0)

      # Convert logits to probabilities using softmax.
      probabilities = ops.softmax(y, axis=-1)

      key, main_key = jax.random.split(main_key)
      sample_token = sampling(
          probabilities[0][sample_index], key
      )

      tokens_generated.append(sample_token)
      start_tokens.append(sample_token)
      num_tokens_generated = len(tokens_generated)

    # Combine the starting tokens with the generated tokens.
    output_tokens = self.start_tokens + tokens_generated
    output_tokens = list(map(int, output_tokens))

    # Decode and print the generated text.
    txt = self.tokenizer.decode(output_tokens)
    print("Generated text:\n", txt, "\n")


class CustomAccuracyPrinter(keras.callbacks.Callback):
  """Custom Keras callback function to print training progress in Lab 3.12.

  Attributes:
    print_every: Print the training progress every `print_every` epochs.
  """

  def __init__(self, print_every: int = 1):
    self.print_every = print_every

  def on_epoch_end(
      self, epoch: int, logs: Optional[Dict[str, Any]] = None
  ) -> None:
    """Prints training and validation metrics at the end of each epoch.

    This function is executed at the end of each epoch. It prints the
    training loss and the validation loss and training and validation
    accuracies, if available. If self.print_every is greater than 1,
    updates are only printed every self.print_every epoch.

    Note that at this stage, learners have not learned the difference between
    validation and test sets and therefore validation loss and validation
    accuracy is renamed to test loss and test accuracy.

    Args:
      epoch: The current epoch number.
      logs: A dictionary containing the current loss and any other metrics that
        were specified when compiling the model.
    """

    if (epoch + 1) % self.print_every != 0:
      return

    if logs is not None:
      log_parts = []
      log_parts.append(f"Epoch {epoch}: Training loss: {logs['loss']:.5f}")
      if "accuracy" in logs:
        log_parts.append(f"training accuracy: {logs['accuracy']*100:.2f}%")
      if "val_loss" in logs:
        log_parts.append(f"test loss: {logs['val_loss']:.5f}")
      if "val_accuracy" in logs:
        log_parts.append(f"test accuracy: {logs['val_accuracy']*100:.2f}%")

      print(", ".join(log_parts))
