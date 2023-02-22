from collections import namedtuple
from random import choice, choices
from copy import deepcopy


# TODO : Change cnn_observation_space, fcn_observation_space, action_space, players
# TODO : fill in all "Your Code Here"


class MarkovState(namedtuple("MarkovStateTemplate", 
field_names=("cnn","fcn","info","turn","terminal","winners"), 
defaults=(
  tuple([tuple([tuple([0 for c in range(13)]) for r in range(13)]) for l in range(7)]),
  tuple([0]), 
  tuple(), 
  0, 
  False, 
  tuple()))):
  # 0 prev prev 0 stones
  # 1 prev 0 stones
  # 2 curr 0 stones
  # 3 All 0 if player 0 turn, 1 if player 1 turn
  # 4 curr 1 stones
  # 5 prev 1 stones
  # 6 prev prev 1 stones

  cnn_observation_space = (7, 13, 13)
  fcn_observation_space = 1
  action_space = 169

  players = 2

  def get_cnn_observation(self):
    """from self.env make cnn input for nn"""
    return self.cnn

  def get_fcn_observation(self):
    """from self.env make fcn input for nn"""
    return self.fcn

  def get_actions(self):
    """Actions that curr agent can take"""
    actions = []
    for r in range(13):
      for c in range(13):
        if self.cnn[2][r][c] == 1 or self.cnn[4][r][c] == 1:
          pass
        else:
          actions.append(r*13+c)
    return actions

  def _get_roll(self, action, all_roll=False):
    """Returns the probabilistic outcome of action"""
    return [None]

  def _step(self, action, roll):
    """Returns the next MrokovState that is the result of action and roll"""
    row, col = action // 13, action % 13

    new_turn = (self.turn+1) % 2
    new_fcn = tuple([new_turn])
    new_info = tuple()

    if self.turn == 0:
      new_cnn0 = tuple([tuple([self.cnn[1][r][c] for c in range(13)]) for r in range(13)])
      new_cnn1 = tuple([tuple([self.cnn[2][r][c] for c in range(13)]) for r in range(13)])
      new_cnn2 = tuple([tuple([1 if row == r and col == c else self.cnn[2][r][c] for c in range(13)]) for r in range(13)])
      new_cnn3 = tuple([tuple([new_turn for c in range(13)]) for r in range(13)])
      new_cnn6 = tuple([tuple([self.cnn[6][r][c] for c in range(13)]) for r in range(13)])
      new_cnn5 = tuple([tuple([self.cnn[5][r][c] for c in range(13)]) for r in range(13)])
      new_cnn4 = tuple([tuple([self.cnn[4][r][c] for c in range(13)]) for r in range(13)])
    else:
      new_cnn0 = tuple([tuple([self.cnn[0][r][c] for c in range(13)]) for r in range(13)])
      new_cnn1 = tuple([tuple([self.cnn[1][r][c] for c in range(13)]) for r in range(13)])
      new_cnn2 = tuple([tuple([self.cnn[2][r][c] for c in range(13)]) for r in range(13)])
      new_cnn3 = tuple([tuple([new_turn for c in range(13)]) for r in range(13)])
      new_cnn6 = tuple([tuple([self.cnn[5][r][c] for c in range(13)]) for r in range(13)])
      new_cnn5 = tuple([tuple([self.cnn[4][r][c] for c in range(13)]) for r in range(13)])
      new_cnn4 = tuple([tuple([1 if row == r and col == c else self.cnn[4][r][c] for c in range(13)]) for r in range(13)])

    new_terminal = False
    new_winners = tuple()
    for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
      count = 1
      for direction in [1, -1]:
        cr, cc = row, col
        while True:
          cr, cc = cr + dr*direction, cc + dc*direction
          if cr in range(13) and cc in range(13):
            if self.turn == 1 and self.cnn[4][cr][cc] == 1:
              count += 1
            elif self.turn == 0 and self.cnn[2][cr][cc] == 1:
              count += 1
            else:
              break
          else:
            break
      if 4 < count:
        new_terminal = True
        new_winners = tuple([self.turn])
        break

    if new_terminal == False:
      no_more_space = True
      for r in range(13):
        for c in range(13):
          if self.cnn[2][r][c] == 0 and self.cnn[4][r][c] == 0:
            no_more_space = False
            break
        if no_more_space == False:
          break
      if no_more_space:
        new_terminal = True
        new_winners = tuple()

    new_cnn = tuple([new_cnn0, new_cnn1, new_cnn2, new_cnn3, new_cnn4, new_cnn5, new_cnn6])

    return MarkovState(new_cnn, new_fcn, new_info, new_turn, new_terminal, new_winners)

  def get_children(self):
    """Return all MrokovState that is the result of action"""
    children = dict()
    for action in self.get_actions():
      rolls = self._get_roll(action, all_roll=True)
      children[action] = {roll: self._step(action, roll) for roll in rolls}
    return children

  def take_action(self, action):
    """get next state that is the result of action"""
    # get next state in stochastic world
    actions = self.get_actions()
    if action not in actions:
      action = choice(actions)
    return self._step(action, self._get_roll(action, all_roll=False))

  def get_turn(self):
    """return turn"""
    return self.turn

  def is_terminal(self):
    """return True if terminal (game over) else False"""
    return self.terminal

  def get_rewards(self):
    """return reward for each player"""
    """
    Reward Clipping
      The rewards obtained from the environment vary in scale depending on the environment. 
      For example, in the Atari game Pong, you score 1 point for each freeze, 
      and in Space Invaders, you score 10 to 30 points for each enemy you defeat.
      
      In DQN, to compensate for this, the parent scale is fixed at -1, 0, and 1 in all environments. 
      This allows learning to be performed using the same hyperparameters regardless of the environment.
        
    because of this reward should be -1, 0 or 1.
    """
    # game over and not draw
    if self.terminal and self.winners:
      return tuple([1 if pidx in self.winners else -1 for pidx in range(MarkovState.players)])
    else:
      return tuple([0 for pidx in range(MarkovState.players)])

  def show(self):
    print(f"====winner:{self.winners}====")
    for r in range(13):
      for c in range(13):
        if self.cnn[2][r][c] == 1:
          print('⚫', end='')
        elif self.cnn[4][r][c] == 1:
          print('⚪', end='')
        else:
          print('__', end='')
      print()
    print("====================")

"""
if __name__ == "__main__":
  curr_state = MarkovState()
  while curr_state.is_terminal() == False:
    curr_state.show()
    action = choice(curr_state.get_actions())
    curr_state = curr_state.take_action(action)
  curr_state.show()
"""
