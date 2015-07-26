require 'nn'
require 'nngraph'
require 'Adam'
require 'base'

params = {
  img_size=64,
  nfeats={64,64},
  conv_layers = 2,
  filtsize = 5,
  poolsize = 2,
  rnn_layers=2,
  rnn_size=200,
  reward_rnn_layers=2,
  reward_rnn_size=100,
  g_size=256,
  a_size=7,
  l_size=4,
  glimpse_x=8,
  glimpse_y=8,
  glimpse_k=3,
  n_channels=3,
  n_glimpses=4,
  glimpse_stdv = .05,
  init_weight=0.8,
  batch_size = 56,
  max_grad_norm = 5,
  n_batches=1
}

local function lstm(x, prev_c, prev_h, input_size, output_size)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(input_size, 4*output_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*output_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates = nn.Reshape(4,output_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function create_network()
  local x  = nn.Identity()()
  local prev_s = nn.Identity()()
  local prev_l = nn.Identity()()

  local nfeats = params.nfeats
  local filtsize = params.filtsize
  local poolsize = params.poolsize
  nfeats[0] = 1

  local x_conv = {}
  x_conv[0] = x

  for i = 1,params.conv_layers do
    local a = nn.SpatialConvolutionMM(nfeats[i-1], nfeats[i], filtsize, filtsize)(x_conv[i-1])
    local b = nn.ReLU()(a)
    x_conv[i] = nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)(b)
  end

  local out_size = nfeats[2]*13*13

  local x_feat = nn.View(out_size)(x_conv[params.conv_layers])

  local g_x = nn.ReLU()(nn.Linear(out_size,params.g_size)(x_feat))
  local g_l = nn.ReLU()(nn.Linear(4,params.g_size)(prev_l))

  local g = nn.ReLU()(nn.CMulTable()({g_x, g_l}))

  local h  = {[0] = g}

  local next_s = {}
  local split = {prev_s:split(2 * params.rnn_layers)}
  for i = 1, params.rnn_layers do
    local prev_c = split[2 * i - 1]
    local prev_h = split[2 * i]
    local next_c, next_h
    if i == 1 then
      next_c, next_h = lstm(h[i - 1], prev_c, prev_h, params.g_size, params.rnn_size)
    else
      next_c, next_h = lstm(h[i - 1], prev_c, prev_h, params.rnn_size, params.rnn_size)
    end
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    h[i] = next_h
  end

  --action nodes

  local a = nn.LogSoftMax()(nn.Linear(params.rnn_size,params.a_size)(h[params.rnn_layers]))

  --location nodes

  local l = nn.Tanh()(nn.Linear(params.rnn_size,4)(h[params.rnn_layers]))

  --baseline

  local b = nn.Sigmoid()(nn.Linear(params.rnn_size,1)(h[params.rnn_layers]))

  local module = nn.gModule({x, prev_l, prev_s},
                                      {a, l, nn.Identity()(next_s),b})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return module
end

