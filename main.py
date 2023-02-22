from os import path
from valuepolicynetwork import ValuePolicyNetwork
from markovstate import MarkovState
from threading import Thread
from mcts import MonteCarloTreeSearch
from torch.utils.data import DataLoader
from random import choices
from torch import nn

import torch
import pickle
import os

L_SEARCH_COUNT = 128
L_SEARCH_DEPTH = 990
L_TEMPERATURE = 10
L_CPUCT = 1
L_GAME_COUNT = 32

EPOCHS = 16

C_SEARCH_COUNT = 128
C_SEARCH_DEPTH = 990
C_TEMPERATURE = 1
C_CPUCT = 1
C_GAME_COUNT = 32

MAX_GAME_LENGHT = 256

player_count = MarkovState.players

def write_data(history, file_name):
  path = './data/{}.history'.format(file_name)
  with open(path, mode='wb') as f:
    pickle.dump(history, f)

def self_play():
  # if game goes on for too long, it will be terminated
  history = []
  best_model = ValuePolicyNetwork()
  best_model.load_state_dict(torch.load('./model/0.pth'))
  game_state = MarkovState()
  for __ in range(MAX_GAME_LENGHT):
    if game_state.is_terminal():
      game_state.show()
      values = game_state.get_rewards()
      for i in range(len(history)):
        history[i][2] = values
      return history
    mcts = MonteCarloTreeSearch(game_state, best_model)
    mcts.learn(L_SEARCH_COUNT, L_SEARCH_DEPTH, L_CPUCT)
    actions, weights = mcts.get_action_weights(L_TEMPERATURE)
    policies = [0 for __ in range(MarkovState.action_space)]
    for action, policy in zip(actions, weights):
      policies[action] = policy
    cnn = game_state.get_cnn_observation()
    fcn = game_state.get_fcn_observation()
    history.append([cnn, fcn, None, policies])
    action = choices(population=actions, weights=weights)[0]
    game_state = game_state.take_action(action)
  return []

def log_history(file_name):
  print('Logging history...')
  history = []
  for i in range(L_GAME_COUNT):
    h = self_play()
    history.extend(h)
  write_data(history, file_name)

def load_data(file_name):
  path = './data/{}.history'.format(file_name)
  with open(path, mode='rb') as f:
    return pickle.load(f)

def train_network(file_name):
  print('Training network...')
  history = load_data(file_name)
  cnns, fcns, values, policies = zip(*history)

  cnn = torch.tensor(cnns, dtype=torch.float32)
  fcn = torch.tensor(fcns, dtype=torch.float32)
  value = torch.tensor(values, dtype=torch.float32)
  policy = torch.tensor(policies, dtype=torch.float32)

  dataset = torch.utils.data.TensorDataset(cnn, fcn, value, policy)

  model = ValuePolicyNetwork()
  model.load_state_dict(torch.load('./model/0.pth'))

  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

  for epoch in range(EPOCHS):
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for i, (cnn, fcn, value, policy) in enumerate(dataloader):
      optimizer.zero_grad()
      v, p = model(cnn, fcn)
      loss = criterion(p, policy) + criterion(v, value)
      loss.backward()
      optimizer.step()
    print('Epoch {} finished'.format(epoch+1))
  torch.save(model.state_dict(), './model/{}.pth'.format(file_name))

def learn(file_name):
  log_history(file_name)
  train_network(file_name)

def competition():
  models = [ValuePolicyNetwork() for __ in range(player_count)]
  for i in range(player_count):
    models[i].load_state_dict(torch.load('./model/{}.pth'.format(i)))
  game_state = MarkovState()
  for __ in range(MAX_GAME_LENGHT):
    if game_state.is_terminal():
      game_state.show()
      return game_state.winners
    mcts = MonteCarloTreeSearch(game_state, models[game_state.get_turn()])
    mcts.learn(C_SEARCH_COUNT, C_SEARCH_DEPTH, C_CPUCT)
    action = mcts.get_action(C_TEMPERATURE)
    game_state = game_state.take_action(action)
  game_state.show()
  return tuple()

def log_competition():
  print('Logging competition...')
  wins = [0 for __ in range(player_count)]
  for i in range(C_GAME_COUNT):
    winners = competition()
    for winner in winners:
      wins[winner] += 1
  print(wins)
  return wins.index(max(wins))

def test():
  best_model = ValuePolicyNetwork()
  best_model.load_state_dict(torch.load('./model/0.pth'))
  game_state = MarkovState()
  for gl in range(MAX_GAME_LENGHT):
    game_state.show()
    if game_state.is_terminal():
      return game_state.winners
    if gl%MarkovState.players == 0:
      valid_actions = game_state.get_actions()
      print('Valid actions: {}'.format(valid_actions))
      while True:
        row, col = map(int, input().split())
        action = row*9+col
        if action in valid_actions:
          break
        print('Invalid action')
      game_state = game_state.take_action(action)
    else:
      mcts = MonteCarloTreeSearch(game_state, best_model)
      mcts.learn(1024, 990, 1)
      action = mcts.get_action(C_TEMPERATURE)
      game_state = game_state.take_action(action)

"""
if __name__ == "__main__":
  print(test())
"""

if __name__ == "__main__":
  os.makedirs('./data/', exist_ok=True)
  os.makedirs('./model/', exist_ok=True)

  if path.exists('./model/0.pth'):
    print('Model already exists')
  else:
    model = ValuePolicyNetwork()
    torch.save(model.state_dict(), './model/0.pth')
    print('Model created')
  
  for __ in range(1000):
    # 1. self play and learn
    threads = [Thread(target=learn, args=(str(i)), daemon=True) for i in range(1, player_count)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    # 2. play against the model
    winner = log_competition()
    # 3. best model selection
    for i in range(player_count):
      if i != winner:
        os.remove('./model/{}.pth'.format(i))
    os.rename('./model/{}.pth'.format(winner), './model/0.pth')
