import csv
import numpy as np
from io import StringIO
import datetime
from PIL import Image
import os


class Model:

  def __init__(self, input_size, latent_size=100):
    self.input_size = input_size

    # Set the size of the latent representation.
    # TODO: How big should this be?
    # self.latent_size = input_size // 2
    self.latent_size = latent_size

    # Initialize the weights
    self.w1 = np.random.uniform(-0.05, 0.05, (input_size + 1, self.latent_size))
    self.w2 = np.random.uniform(-0.05, 0.05, (self.latent_size + 1, input_size))

    self.eta = 0.1
    self.log = []

  def set_learning_rate(self, eta):
    self.eta = eta

  def train(self, data, test_data, label, num_epochs=10, momentum=0, batch_size=100):

    m = data.shape[0]
    data = data.copy()

    prev_up_latent = np.zeros_like(self.w1)
    prev_up_recon = np.zeros_like(self.w2)


    # save original test images
    for i in range(6):
      save_image(test_data[i], 32, f"images/{label}/0-{i}.png")


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

      # log the error for training and test data
      _, actual = self.forward(data)
      _, test_actual = self.forward(test_data)
      self.log.append({
        "epoch": epoch,
        "train_error": err(data, actual),
        "test_error": err(test_data, test_actual)
      })

      # save in-progress images
      for i in range(6):
        save_image(test_actual[i], 32, f"images/{label}/{epoch+1}-{i}.png")

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

def err(expected, actual):
  diff = expected - actual
  normError = 0
  for row in diff:
      normError += np.linalg.norm(row)

  return normError / actual.shape[0]

def save_image(y, side, filepath):
  y = y.reshape((side,side))
  y = y * 255
  y_im = Image.fromarray(y)
  y_im.convert('RGB').save(filepath)


def load_fish():
  """Returns the preprocessed fish images as a numpy array."""

  with open("preprocessor/FishConverter/data.csv") as f:
    data = StringIO()
    # remove trailing commas
    for line in f.readlines():
      if len(line) > 10:
        data.write(line[:-2])
        data.write("\n")
    # reset cursor to beginning of buffer
    data.seek(0)
    return np.loadtxt(data, delimiter=",") / 255


if __name__ == "__main__":

  momentum = 0.5
  num_epochs = 300
  batch_size = 100

  data = load_fish()
  train_size = 3200
  train_data = data[:train_size]
  test_data = data[train_size:]

  start_time = datetime.datetime.now()

  for momentum in [0, 0.5, 0.8]:

    for learning_rate in [0.01, 0.05, 0.1]:

      label = str(start_time) + f"-lr{learning_rate}-mom{momentum}"
      os.mkdir(f"images/{label}")

      model = Model(32 * 32, latent_size=100)
      model.set_learning_rate(learning_rate)
      model.train(train_data, test_data, label, num_epochs=num_epochs, momentum=momentum, batch_size=batch_size)

      with open(f"results-{label}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["learning rate", "momentum", "batch size", "epoch", "train error", "test error"])

        for entry in model.log:
          row = [learning_rate, momentum, batch_size, entry["epoch"], entry["train_error"], entry["test_error"]]
          writer.writerow(row)