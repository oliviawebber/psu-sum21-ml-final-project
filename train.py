import numpy as np



class Model:

  def __init__(self, input_size):
    self.input_size = input_size

    # Set the size of the latent representation.
    # TODO: How big should this be?
    # self.latent_size = input_size // 2
    self.latent_size = 20

    # Initialize the weights
    self.w1 = np.random.uniform(-0.05, 0.05, (input_size + 1, self.latent_size))
    self.w2 = np.random.uniform(-0.05, 0.05, (self.latent_size + 1, input_size))

    self.eta = 0.1

  def set_learning_rate(self, eta):
    self.eta = eta

  def train(self, data, num_epochs=10, momentum=0, batch_size=100):

    m = data.shape[0]
    data = data.copy()

    prev_up_latent = np.zeros_like(self.w1)
    prev_up_recon = np.zeros_like(self.w2)

    for epoch in range(num_epochs):

      print(f"starting epoch {epoch + 1}")

      # shuffle before getting batches
      np.random.shuffle(data)

      for i in range(m // batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data = data[start:end, :]

        latent, recon = self.forward(batch_data)
        up_latent, up_recon = self.backward(batch_data, latent, recon)

        rate = self.eta * (batch_size / m)
        self.w1 += rate * (up_latent + (momentum * prev_up_latent))
        self.w2 += rate * (up_recon + (momentum * prev_up_recon))
        prev_up_latent = up_latent
        prev_up_recon = up_recon

  def forward(self, data):

    # m is the number of training instances
    data = with_bias(data)

    latent_activations = data @ self.w1
    latent_output = sigmoid(latent_activations)
    latent = with_bias(latent_output)

    reconstruction_activations = latent @ self.w2
    reconstruction = sigmoid(reconstruction_activations)

    return latent, reconstruction

  def backward(self, expected, latent, reconstruction):
    err_reconstruction = reconstruction * (1 - reconstruction) * (expected - reconstruction)
    err_latent = latent * (1 - latent) * (err_reconstruction @ self.w2.T)

    update_w2 = latent.T @ err_reconstruction
    update_w1 = with_bias(expected).T @ err_latent[:,:-1]

    return update_w1, update_w2

  def run(self, input):
    """Run autoencoder on a single input."""
    _, recon = self.forward(np.array([input]).reshape((1,self.input_size)))
    return recon


def with_bias(data, bias=1):
  """Adds a bias to an array"""
  return np.append(data, np.full((data.shape[0], 1), bias), axis=1)


def sigmoid(data):
  """Sigmoid activation function"""
  return 1 / (1 + np.exp(-1 * data))

