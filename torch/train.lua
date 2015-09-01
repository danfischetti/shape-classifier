require 'model'
require 'image'
json = require('json')
require('json.rpc')

local filepath = '/ramcache/renders/'

model={}

function transfer_data(x)
  return x:cuda()
end

local function argmax(x)
  local max_val = torch.max(x)
   for i = 1, x:size(1) do
      if x[i] == max_val then
         return i
      end
   end
end

function get_classes(path)
  class_labels = {}
  class_labels['vehicle']=1
  class_labels['animal']=2
  class_labels['household']=3
  class_labels['building']=4
  class_labels['furniture']=5
  class_labels['plant']=6
  class_labels['other']=7
  classes = {}
  file = io.open(path, "r")
  current_class = ""
  for line in file:lines() do
    n = tonumber(line)
    if n~=nil then
      classes[n]=class_labels[current_class]
    elseif line~=nil then
      current_class=line
    end
  end
  file.close(file)
  return classes
end  

function get_view(l,index,n)
  json.rpc.call('http://localhost:9090','get_view',{l[1],l[2],l[3],l[4],index,n})
  return image.load(filepath .. 'm' .. n .. '.png')
end

function loadBatch(n)
  json.rpc.call('http://localhost:9090','loadObjects',{n})
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

local function fp(indices,y)
  g_replace_table(model.s[0], model.start_s)
  model.l[0]:copy(model.start_l)
  model.l_mean[0]:copy(model.start_l)
  local x = torch.zeros(params.batch_size,1,params.img_size,params.img_size)
  for i = 1, params.n_glimpses do
    for j = 1,params.batch_size do
      x[j]:copy(get_view(model.l[i-1][j],indices[j],j))
    end
    model.x[i]=transfer_data(x)
    model.a, model.l_mean[i], model.s[i], model.b = 
      unpack(model.networks[i]:forward({model.x[i], model.l[i-1], model.s[i-1]}))
    local epsilon = transfer_data(torch.randn(model.l[i]:size()))
    model.l[i]:add(model.l_mean[i],params.glimpse_stdv,epsilon)
  end
  model.err = model.criterion:forward(model.a,y)
  for i = 1,params.batch_size do
    --print(y[i] .. ' ' .. argmax(model.a[i]))
    if y[i] == argmax(model.a[i]) then
      model.reward[i] = 1
    else 
      model.reward[i] = 0
    end
  end
  return {model.a,model.err}
end

local function bp(y)
  paramdx:zero()
  reset_ds()
  local l = model.l[params.n_glimpses - 1]
  local s = model.s[params.n_glimpses - 1]
  local x = model.x[params.n_glimpses]
  local l_err = transfer_data(torch.zeros(l:size()))
  local a_err = model.criterion:backward(model.a,y)
  local b_err = transfer_data((model.b-model.reward)*2)
  local s_err = model.networks[params.n_glimpses]:backward({x,l,s},{a_err,l_err,model.ds,b_err})[3]
  a_err:zero()
  b_err:zero()
  g_replace_table(model.ds, s_err)
  cutorch.synchronize()
  for i = params.n_glimpses-1, 1, -1 do
    l = model.l[i-1]
    x = model.x[i]
    l_mean = model.l_mean[i-1]
    s = model.s[i - 1]
    l_err[{{},1}]:cmul(-(model.reward-model.b),(l_mean-l)[{{},1}]*2/(params.glimpse_stdv*params.batch_size))
    l_err[{{},2}]:cmul(-(model.reward-model.b),(l_mean-l)[{{},2}]*2/(params.glimpse_stdv*params.batch_size))
    l_err[{{},3}]:cmul(-(model.reward-model.b),(l_mean-l)[{{},3}]*2/(params.glimpse_stdv*params.batch_size))
    l_err[{{},4}]:cmul(-(model.reward-model.b),(l_mean-l)[{{},4}]*2/(params.glimpse_stdv*params.batch_size))

    s_err = model.networks[i]:backward({x,l,s},{a_err,l_err,model.ds,b_err})[3]
    g_replace_table(model.ds, s_err)
    cutorch.synchronize()
  end
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
end

local function init()
  local core_network = transfer_data(create_network())
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.l = {}
  model.b = {}
  model.x = {}
  model.l_mean = {}
  model.ds = {}
  model.start_s = {}
  model.start_l = transfer_data(torch.zeros(params.batch_size, 4))
  for j = 0, params.n_glimpses do
    model.s[j] = {}
    model.l[j] = transfer_data(torch.zeros(params.batch_size, 4))
    model.l_mean[j] = transfer_data(torch.zeros(params.batch_size, 4))
    for d = 1, 2 * params.rnn_layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.rnn_layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.criterion = transfer_data(nn.ClassNLLCriterion())
  model.networks = g_cloneManyTimes(core_network, params.n_glimpses)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(1))
  model.reward = transfer_data(torch.zeros(params.batch_size))
  model.classes = get_classes('../princeton_blend/classes.txt')
  print("Model built.")
end

function validate()
  local err = 0
  local class_right = 0
  local class_total = 0
  for i = 1,4 do
    local j = params.n_batches + i
    loadBatch(j)
    local y = torch.zeros(params.batch_size)
    local indices = {}
    for k = 1,params.batch_size do
      index=4*(j-1)+((k-1)%14)*100+math.floor((k-1)/14)
      indices[k]=index
      y[k]=model.classes[index]
    end
    y = transfer_data(y)
    local out = fp(indices,y)
    local labels = out[1]
    for k = 1,params.batch_size do
      class_total = class_total + 1
      if argmax(labels[k]) == y[k] then
        class_right = class_right + 1
      end
    end
    err = err+out[2]
  end
  print('Validation err: ' .. err)
  print(class_right .. '/' .. class_total .. ' classified correctly')
end

function main()
  init()
  local adam = nn.Adam(paramx,paramdx)
  for i = 1,100 do
    local err = 0
    local class_right = 0
    local class_total = 0
    for j = 1,params.n_batches do
      local y = torch.zeros(params.batch_size)
      local indices = {}
      local j2 = j
      if j == 16 then
        j2 = 25
      end
      for k = 1,params.batch_size do
        index=4*(j2-1)+((k-1)%14)*100+math.floor((k-1)/14)
        indices[k]=index
        y[k]=model.classes[index]
      end
      y = transfer_data(y)
      loadBatch(j2)
      local out = fp(indices,y)
      local labels = out[1]
      for k = 1,params.batch_size do
        class_total = class_total + 1
        if argmax(labels[k]) == y[k] then
          class_right = class_right + 1
        end
      end
      err = err+out[2]
      bp(y)
      adam:step()
    end
    print('Epoch ' .. i .. ', Train err: ' .. err)
    validate()
  end
end

main()