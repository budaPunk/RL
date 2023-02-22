from torch import nn, cat
from markovstate import MarkovState


class ResidualLayer(nn.Module):
  def __init__(self, o_channels=256, h_channels=256):
    super(ResidualLayer, self).__init__()
    # batch, o_channels, height, width -> batch, h_channels, height, width
    self.conv1 = nn.Conv2d(in_channels=o_channels, out_channels=h_channels, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(num_features=h_channels)
    self.elu1 = nn.ELU()
    # batch, h_channels, height, width -> batch, o_channels, height, width
    self.conv2 = nn.Conv2d(in_channels=h_channels, out_channels=o_channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(num_features=o_channels)
    # Skip connection
    self.shortcut = nn.Sequential()
    self.elu2 = nn.ELU()

  def forward(self, x):
    # batch, o_channels, height, width -> batch, h_channels, height, width
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.elu1(out)
    # batch, h_channels, height, width -> batch, o_channels, height, width
    out = self.conv2(out)
    out = self.bn2(out)
    # Skip connection
    out += self.shortcut(x)
    out = self.elu2(out)
    return out


class ValuePolicyNetwork(nn.Module):
  """
  ========== ========== ========== ========== ========== ========== ========== ==========
  cnnInput                    fcnInput
  (b, d, h, w)                (b, w)
  (b, cnn_final)              (b, fcn_final)
          (b, cnn_final+fcn_final)
  (b, 1)                      (b, action_space)
  value out                   policy out
  ========== ========== ========== ========== ========== ========== ========== ==========
  """
  def __init__(self, cnn_hidden=256, fcn_hidden=256, cnn_final=128, fcn_final=128):
    super(ValuePolicyNetwork, self).__init__()

    # ========== ========== ========== ==========
    # in1 = MarkovState.cnn_observation_space
    # ========== ========== ========== ==========
    self.dhw = MarkovState.cnn_observation_space
    # batch, depth, height, width -> batch, cnn_hidden, height, width
    self.cnn_input = nn.Sequential(
      # batch, depth, height, width -> batch, cnn_hidden, height, width
      nn.Conv2d(in_channels=self.dhw[0], out_channels=cnn_hidden, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(num_features=cnn_hidden),
      nn.ELU()
    )

    _layers = []
    for __ in range(3):
      _layers.append(ResidualLayer(o_channels=cnn_hidden, h_channels=256))
    # batch, cnn_hidden, height, width -> batch, cnn_hidden, height, width
    self.residual = nn.Sequential(*_layers)

    # batch, cnn_hidden, height, width -> batch, cnn_final
    self.line = nn.Sequential(
      # batch, cnn_hidden, height, width -> batch, 1, height, width
      nn.Conv2d(in_channels=cnn_hidden, out_channels=1, kernel_size=1, stride=1, padding=0),
      # batch, 1, height, width -> batch, height*width
      nn.Flatten(),
      nn.BatchNorm1d(num_features=self.dhw[1]*self.dhw[2]),
      nn.ELU(),
      # batch, height*width -> batch, cnn_final
      nn.Linear(in_features=self.dhw[1]*self.dhw[2], out_features=cnn_final)
    )

    # ========== ========== ========== ==========
    # in2 = MarkovState.fcn_observation_space
    # ========== ========== ========== ==========
    # batch, MarkovState.fcn_observation_space -> batch, fcn_final
    self.fcn_input = nn.Sequential(
      # batch, MarkovState.fcn_observation_space -> batch, fcn_hidden
      nn.Linear(in_features=MarkovState.fcn_observation_space, out_features=fcn_hidden),
      # batch, fcn_hidden -> batch, fcn_hidden
      nn.Linear(in_features=fcn_hidden, out_features=fcn_hidden),
      nn.BatchNorm1d(num_features=fcn_hidden),
      nn.ELU(),
      # batch, fcn_hidden -> batch, fcn_final
      nn.Linear(in_features=fcn_hidden, out_features=fcn_final)
    )

    # ========== ========== ========== ==========
    # out1 = 1
    # ========== ========== ========== ==========
    # batch, cnn_final+fcn_final -> batch, MarkovState.players
    self.value_head = nn.Sequential(
      nn.BatchNorm1d(num_features=cnn_final+fcn_final),
      nn.ELU(),
      # batch, cnn_final+fcn_final -> batch, 256
      nn.Linear(in_features=cnn_final+fcn_final, out_features=256),
      nn.ELU(),
      # scalar
      # batch, 256 -> batch, MarkovState.players
      nn.Linear(in_features=256, out_features=MarkovState.players),
      # game value for current player [-1, 1]
      nn.Tanh()
    )

    # ========== ========== ========== ==========
    # out2 = MarkovState.action_space
    # ========== ========== ========== ==========
    # batch, cnn_final+fcn_final -> batch, action_space
    self.policy_head = nn.Sequential(
      nn.BatchNorm1d(num_features=cnn_final+fcn_final),
      nn.ELU(),
      # batch, cnn_final+fcn_final -> batch, action_space
      nn.Linear(in_features=cnn_final+fcn_final, out_features=MarkovState.action_space)
    )

  def forward(self, cnn=None, fcn=None):
    # PyTorch multiple input
    # https://stackoverflow.com/questions/51700729/how-to-construct-a-network-with-two-inputs-in-pytorch
    # https://stackoverflow.com/questions/73266661/multiple-input-model-with-pytorch-input-and-output-features

    # batch, depth, height, width -> batch, cnn_final
    cnn_result = self.cnn_input(cnn)
    residual_result = self.residual(cnn_result)
    line_result = self.line(residual_result)

    # batch, fcn_observation_space -> batch, fcn_final
    fcn_result = self.fcn_input(fcn)

    # batch, cnn_final + batch, fcn_final -> batch, cnn_final+fcn_final
    combined = cat((line_result.view(line_result.size(0), -1),
                    fcn_result.view(fcn_result.size(0), -1)), dim=1)

    # PyTorch multiple output
    # https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440

    # batch, cnn_final+fcn_final -> batch, players
    value_result = self.value_head(combined)

    # batch, cnn_final+fcn_final -> batch, action_space
    policy_result = self.policy_head(combined)

    return value_result, policy_result


if __name__ == "__main__":
  import torch
  # test ValuePolicyNetwork
  v = ValuePolicyNetwork()

  # for training
  cnn_test = torch.randn(64, 7, 13, 13)
  fcn_test = torch.randn(64, 13)
  print(cnn_test.shape, fcn_test.shape)
  v.train()
  value, policy = v(cnn=cnn_test, fcn=fcn_test)
  print(value.shape, policy.shape)
  
  # for evaluation
  cnn_test = torch.randn(1, 7, 13, 13)
  fcn_test = torch.randn(1, 13)
  print(cnn_test.shape, fcn_test.shape)
  v.eval()
  value, policy = v(cnn=cnn_test, fcn=fcn_test)
  print(value.shape, policy.shape)