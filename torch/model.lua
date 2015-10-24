require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
cudnn.benchmark = true
require 'nngraph'
require 'Adam'
require 'base'

params = {
  img_size=64,
  nfeats={64,96,128},
  out_size = 128*4*4,
  conv_layers = 3,
  filtsize = 5,
  poolsize = 2,
  rnn_layers=2,
  fc_layers=1,
  rnn_size=256,
  g_size=256,
  a_size=7,
  l_size=4,
  glimpse_x=8,
  glimpse_y=8,
  glimpse_k=3,
  n_glimpses=2,
  glimpse_stdv=.15,
  init_weight=0.05,
  batch_size = 56,
  max_grad_norm = 25,
  n_batches=16
}

local function rnn(x, prev_h, input_size, output_size)
  local i2h = nn.Linear(input_size, output_size)
  local h2h = nn.Linear(params.rnn_size, output_size)

  local next_h = nn.Tanh()(nn.CAddTable()({i2h(x), h2h(prev_h)}))

  return i2h, h2h, next_h
end

local function lstm(x, prev_c, prev_h, input_size, output_size)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(input_size, 4*output_size)
  local h2h = nn.Linear(params.rnn_size, 4*output_size)
  local gates = nn.CAddTable()({i2h(x), h2h(prev_h)})
  
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

  return i2h, h2h, next_c, next_h
end

function core_network()
  local x  = nn.Identity()()
  local prev_s = nn.Identity()()
  local prev_l = nn.Identity()()

  local nfeats = params.nfeats
  local filtsize = params.filtsize
  local poolsize = params.poolsize
  nfeats[0] = 1

  local x_conv = {}
  x_conv[0] = x

  local conv_transforms = {}

  local t = {}

  for i = 1,params.conv_layers do
    conv_transforms[i] = cudnn.SpatialConvolution(nfeats[i-1], nfeats[i], filtsize, filtsize)
    local a = conv_transforms[i](x_conv[i-1])
    local b = nn.ReLU(true)(a)
   -- conv_transforms[2+2*(i-1)] = cudnn.SpatialConvolution(nfeats[i], nfeats[i], filtsize, filtsize)
    --local c = conv_transforms[2+2*(i-1)](b)
    --local d = nn.ReLU(true)(c)
    x_conv[i] = cudnn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)(b)
  end

  --local out_size = nfeats[2]*13*13

  local x_feat = nn.View(params.out_size)(x_conv[params.conv_layers])

  local g_x = nn.ReLU(true)(nn.Linear(params.out_size,params.g_size)(x_feat))
  local g_l = nn.ReLU(true)(nn.Linear(4,params.g_size)(prev_l))

  local transforms = {}
  transforms[0] = nn.Linear(2*params.g_size,params.g_size)
  transforms[0].weight:normal(0, 2*params.init_weight)

  local g = nn.ReLU(true)(transforms[0](nn.JoinTable(2,2)({g_x, g_l})))

  --transforms[0] = nn.Linear(params.g_size,params.g_size)
  --transforms[0].weight:normal(0, 2*params.init_weight)

  --local g = nn.ReLU(true)(transforms[0](g_x))


  for i = 1,params.fc_layers do
    transforms[i] = nn.Linear(2*params.g_size,params.g_size)
    --local t = nn.Sigmoid()(transforms[i](g))
    --local c = nn.MulConstant(-1,true)(nn.AddConstant(-1,true)(t))
    local h = nn.ReLU(true)(transforms[i](nn.JoinTable(2,2)({g, g_l})))
    --g = nn.CAddTable()({nn.CMulTable()({t,h}),nn.CMulTable()({c,g})})
    g = h
  end

  local h  = {[0] = nn.JoinTable(2,2)({g, g_l})}

  local next_s = {}
  --local split = {prev_s:split(2 * params.rnn_layers)}
  local split
  if params.rnn_layers == 1 then
    split = prev_s
  else
    split = {prev_s:split(params.rnn_layers)}
  end
  local i2h = {}
  local h2h = {}
  for i = 1, params.rnn_layers do
    --local prev_c = split[2 * i - 1]
    --local prev_h = split[2 * i]
    local prev_h = split[i]
    --local next_c, next_h
    local next_h
    if i == 1 then
      --i2h[i], h2h[i], next_c, next_h = lstm(h[i - 1], prev_c, prev_h, params.g_size, params.rnn_size)
      i2h[i], h2h[i], next_h = rnn(h[i - 1], prev_h, 2*params.g_size, params.rnn_size)
    else
      --i2h[i], h2h[i], next_c, next_h = lstm(h[i - 1], prev_c, prev_h, params.rnn_size, params.rnn_size)
      i2h[i], h2h[i], next_h = rnn(h[i - 1], prev_h, params.rnn_size, params.rnn_size)
    end
    --table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    h[i] = next_h
  end

  local module = nn.gModule({x, prev_l, prev_s},
                                      {h[params.rnn_layers],x_feat, nn.Identity()(next_s)})
  module:getParameters():normal(0, params.init_weight)
  for i = 1,params.conv_layers do
    local init1 = 60/(nfeats[i-1]*params.filtsize*params.filtsize)
    --local init2 = 60/(nfeats[i]*params.filtsize*params.filtsize)
    conv_transforms[i].weight:normal(0,init1)
    conv_transforms[i].bias:normal(0, init1/120)
   --conv_transforms[2+2*(i-1)].weight:normal(0,init2)
    --conv_transforms[2+2*(i-1)].bias:normal(0, init2/120)
  end
  for i = 0,params.fc_layers do
    transforms[i].bias:normal(0, params.init_weight/20)
  end
  for i = 1,params.rnn_layers do
    i2h[i].weight:normal(0,params.init_weight)
    h2h[i].weight:normal(0,params.init_weight)
    i2h[i].bias:normal(0,params.init_weight/10)
    h2h[i].bias:normal(0,params.init_weight/10)
  end
  return module
end

function pred_network()
  local net = nn.Sequential()
  local p_transform = nn.Linear(params.rnn_size,params.a_size)
  net:add(p_transform)
  net:add(cudnn.LogSoftMax())
  net:getParameters():normal(0,params.init_weight/10)
  p_transform.bias:normal(0,params.init_weight/20)
  return net
end

function location_network()
  local net = nn.Sequential()
  local l_transform = nn.Linear(params.rnn_size,4)
  net:add(l_transform)
  net:add(nn.HardTanh())
  net:getParameters():normal(0, params.init_weight)
  l_transform.bias[3] = 0.5
  return net
end

function bias_network()
  local net = nn.Sequential()
  local b_transform = nn.Linear(1,1)
  b_transform.bias[1]=0.2
  net:add(b_transform)
  return net
end

function reconstruct_network()
  local nfeats = params.nfeats
  local filtsize = params.filtsize
  local poolsize = params.poolsize
  local net = nn.Sequential()
  local transforms = {}
  net:add(nn.View(nfeats[3],4,4))
  net:add(nn.SpatialUpSamplingNearest(3))
  net:add(nn.SpatialZeroPadding(2,2,2,2))
  transforms[1] = cudnn.SpatialConvolution(nfeats[3], nfeats[2], filtsize, filtsize)
  net:add(transforms[1])
  net:add(nn.ReLU(true))
  net:add(nn.SpatialUpSamplingNearest(3))
  transforms[2]=cudnn.SpatialConvolution(nfeats[2], nfeats[1], filtsize, filtsize)
  net:add(transforms[2])
  net:add(nn.ReLU(true))
  transforms[3] = cudnn.SpatialConvolution(nfeats[1], 1, 1, 1)
  net:add(transforms[3])
  net:add(nn.Sigmoid(true))
  for i = 1,3 do
    transforms[i].weight:normal(0,params.init_weight/10)
    transforms[i].bias:zero()
  end
  return net
end

