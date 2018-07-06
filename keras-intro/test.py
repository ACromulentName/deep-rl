import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import pickle

def test(model, test_data):
  obs = test_data['observations']
  actions = test_data['actions']
  returns = test_data['returns']
  score = model.evaluate(obs, actions, batch_size=128)
  print ("Score: ", score)

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('test_data', type=str)
  parser.add_argument('model_path', type=str)
  args = parser.parse_args()

  # Load data (deserialize)
  with open(args.test_data, 'rb') as handle:
    test_data = pickle.load(handle)

  model = load_model(args.model_path)
  test(model, test_data)

if __name__ == "__main__":
  main()
