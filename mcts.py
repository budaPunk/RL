from valuepolicynetwork import ValuePolicyNetwork
from markovstate import MarkovState
from random import choice, choices

import numpy as np
import torch


class Node:
  def __init__(self, state:MarkovState):
    self._state = state
    # N    = {State: VisitCount, ...}
    # W    = {State: SumValueOfNextState, ...}
    # P    = {State: {Action: ProbabilityOfSelectingAction, ...}, ...}
    # tree = {Action: {Roll: C_Node, ...}, ...}
    self._W = 0
    self._N = 0
    self._P = dict()
    self._children = dict()
  
  def expand(self, policies:list):
    curr_state = self._state
    self._children = curr_state.get_children()
    for action in self._children.keys():
      self._P[action] = policies[action]
      for roll in self._children[action].keys():
        next_state = self._children[action][roll]
        self._children[action][roll] = Node(next_state)

def predict(curr_state:MarkovState, model:ValuePolicyNetwork):
  # convert state to input tensor
  cnn = torch.tensor([curr_state.get_cnn_observation()], dtype=torch.float32)
  fcn = torch.tensor([curr_state.get_fcn_observation()], dtype=torch.float32)

  # predict
  model.eval()
  v, p = model(cnn, fcn)
  return v.tolist()[0], p.tolist()[0]

def select(curr_node:Node, cpuct:float, p=False):
  max_action, max_action_value = [], -np.inf
  for action in curr_node._children.keys():
    action_visits, action_wins = 0, 0
    for roll in curr_node._children[action].keys():
      child_node = curr_node._children[action][roll]
      action_visits += child_node._N
      action_wins += child_node._W
    action_winrate = action_wins / action_visits if action_visits else 0
    action_prior = curr_node._P[action]
    bias = np.sqrt(curr_node._N) / (1 + action_visits)
    if p:
      print(action, action_visits, action_winrate, bias, action_winrate, cpuct * action_prior * bias)
    action_value = action_winrate + cpuct * action_prior * bias
    if action_value > max_action_value:
      max_action_value = action_value
      max_action = [action]
    elif action_value == max_action_value:
      max_action.append(action)
  return choice(max_action)

def search(curr_node:Node, model:ValuePolicyNetwork, depth:int, cpuct:float):
  curr_state = curr_node._state
  game_over = curr_state.is_terminal()
  if game_over == True:
    values = curr_state.get_rewards()
    curr_node._N += 1
    curr_node._W += values[curr_state.get_turn()]
    return values
  elif game_over == False:
    if curr_node._children:
      if depth == 0:
        values, _ = predict(curr_state, model)
      else:
        max_action = select(curr_node, cpuct)
        next_state = curr_state.take_action(max_action)
        for roll in curr_node._children[max_action].keys():
          if next_state == curr_node._children[max_action][roll]._state:
            next_node = curr_node._children[max_action][roll]
            break
        values = search(next_node, model, depth-1, cpuct)
      curr_node._N += 1
      curr_node._W += values[curr_state.get_turn()]
      return values
    else:
      values, policies = predict(curr_state, model)
      curr_node.expand(policies)
      curr_node._N += 1
      curr_node._W += values[curr_state.get_turn()]
      return values
  else:
    raise Exception('MarkovState.is_terminal() return None')

class MonteCarloTreeSearch:
  def __init__(self, root_state:MarkovState, model:ValuePolicyNetwork):
    self._root_node = Node(root_state)
    self._model = model
  
  def learn(self, iterations:int, depth:int, cpuct:float):
    for _ in range(iterations):
      search(self._root_node, self._model, depth, cpuct)
  
  def _get_actions_visits(self):
    if not self._root_node._children:
      raise Exception('MCTS has not been trained.')
    
    visits = dict()
    for action in self._root_node._children.keys():
      visit = 0
      for roll in self._root_node._children[action].keys():
        visit += self._root_node._children[action][roll]._N
      visits[action] = visit
    return visits

  def get_action_weights(self, temperature:float=1.0):
    if not self._root_node._children:
      raise Exception('MCTS has not been trained.')
    
    visits = self._get_actions_visits()
    if temperature == 0:
      max_visit = max(visits.values())
      weights = [1 if visit == max_visit else 0 for visit in visits.values()]
    else:
      weights = [visit ** (1 / temperature) for visit in visits.values()]
      weights = [weight / sum(weights) for weight in weights]
    return list(visits.keys()), weights
  
  def get_action(self, temperature:float=1.0):
    actions, weights = self.get_action_weights(temperature)
    return choices(actions, weights)[0]

"""
if __name__ == "__main__":
  game_length = {0:0, 1:0, 2:0, 3:0, 4:0}
  for __ in range(100):
    model = ValuePolicyNetwork()
    state = MarkovState()
    for gl in range(500): 
      if state.is_terminal():
        game_length[gl//100] += 1
        break
      state.show()
      print(state.get_actions())
      mcts = MonteCarloTreeSearch(state, model)
      mcts.learn(int(input()), 990, 100)
      select(mcts._root_node, 0.5, True)
      action = mcts.get_action(temperature=10000)
      state = state.take_action(action)
  print(game_length)
"""