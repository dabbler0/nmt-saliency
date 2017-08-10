function append_table(dest, inp)

  for i=1,#inp do
    table.insert(dest, inp[1])
  end

  return dest
end

function slice_table(src, index)
  local result = {}
  for i=index,#src do
    table.insert(result, src[i])
  end

  return result
end

function printMemProfile()
  free, total = cutorch.getMemoryUsage()
  print('GPU MEMORY FREE: ', free, 'of', total)
end

local first_hidden = {}
local sequence_inputs = {}
local input_relevances = torch.CudaTensor()
local true_final = torch.CudaTensor()
local initial_relevances = {}

function LRP_saliency(
    alphabet,
    model,
    normalizer,
    sentence,
    modified)

  local opt, encoder_clones, lookup = model.opt, model.clones, model.lookup

  -- Construct beginning hidden state
  for i=1,2*opt.num_layers do
    first_hidden[i] = first_hidden[i] or torch.CudaTensor()
    first_hidden[i]:resize(opt.rnn_size, opt.rnn_size):zero()
  end
  for i=2*opt.num_layers+1,#first_hidden do
    first_hidden[i] = nil
  end

  -- Forward pass
  local rnn_state = first_hidden
  for t=1,#sentence do

    sequence_inputs[t] = sequence_inputs[t] or torch.CudaTensor()
    sequence_inputs[t]:resize(1, opt.rnn_size)
    sequence_inputs[t]:copy(
      lookup:forward(torch.CudaTensor{sentence[t]})
    )
    sequence_inputs[t] = sequence_inputs[t]:expand(opt.rnn_size, opt.rnn_size)

    local encoder_input = {sequence_inputs[t]}
    append_table(encoder_input, rnn_state)
    rnn_state = encoder_clones[t]:forward(encoder_input)
  end

  -- Relevance
  for i=1,2*opt.num_layers do
    initial_relevances[i] = initial_relevances[i] or torch.CudaTensor()
    initial_relevances[i]:resize(opt.rnn_size, opt.rnn_size):zero()
  end

  true_final:resizeAs(rnn_state[#rnn_state][1]):
    copy(rnn_state[#rnn_state][1]):
    cdiv(normalizer[2][1])
  initial_relevances[2*opt.num_layers]:zero():diag(true_final)

  input_relevances:resize(#sentence, opt.rnn_size):zero()

  local relevance_state = initial_relevances
  for t=#sentence,1,-1 do
    relevance_state = LRP(encoder_clones[t], relevance_state, modified)

    -- The input relevance state should now be a 500x500 vector representing
    -- total relevance over the word embedding. Summing over the second
    -- dimension will get us the desired relevances.
    input_relevances:narrow(1, t, 1):sum(relevance_state[1], 2)
    relevance_state = slice_table(relevance_state, 2)
  end

  local affinities = {}
  for i=1,#sentence do
    affinities[i] = input_relevances[i]
  end

  return affinities
end

function get_max(tensor)
  local coordinates = {}
  tensor = torch.abs(tensor)
  -- Iterate our own darn self over this tensor 'cos Torch doesn't have any kind of useful max
  overall_max = torch.max(tensor)
  local size = tensor:size()

  coordinates = {}

  for i=1,#size do
    for j=1,size[i] do
      if tensor:narrow(i, j, 1):max() == overall_max then
        coordinates[i] = j
      end
    end
  end

  return {tensor[coordinates], coordinates}
end

function LRP(gmodule, R, modified)
  local relevances = {}

  -- Topological sort of nodes in the
  -- gmodule
  local toposorted = {}
  local visited = {}

  function dfs(node)
    if visited[node] then
      return
    end
    visited[node] = true

    for dependency, t in pairs(node.data.reverseMap) do
      dfs(dependency)
    end

    table.insert(toposorted, node.data)
  end

  dfs(gmodule.innode)

  -- PROPAGATION
  relevances[gmodule.outnode.data] = R

  for i=1,#toposorted do
    local node = toposorted[i]
    local relevance = relevances[node]

    --print(torch.typename(node.module))
    --[[
    if node.module ~= nil then
      if type(relevance) == 'table' then
        for i=1,#relevance do
          print('Initial 316', i, relevance[i][316][316])
        end
      elseif relevance:size(2) == 2000 then
        print('Initial 316', relevance[316][1316])
      elseif relevance:size(2) == 4 then
        print('Initial 316', relevance[316][3][316])
      else
        print('Initial 316', relevance[316][316])
      end
    end
    ]]
    local input_relevance = relevance_propagate(node, relevance, modified)
    --[[
    if node.module ~= nil then
      if type(input_relevance) == 'table' then
        for i=1,#input_relevance do
          if input_relevance[i]:size(2) == 2000 then
            print('Result 316', i, input_relevance[i][316][1316], node.input[i][316][1316])
          else
            print('Result 316', i, input_relevance[i][316][316], node.input[i][316][316])
          end
        end
      elseif input_relevance:size(2) == 2000 then
        print('Result 316', input_relevance[316][1316], node.input[1][316][1316])
      elseif input_relevance:size(2) == 4 then
        print('Result 316', input_relevance[316][3][316], node.input[1][316][3][316])
      else
        print('Result 316', input_relevance[316][316], node.input[1][316][316])
      end
    end
    ]]
    --[[
    if node.module ~= nil then
      if type(input_relevance) == 'table' then
        print(node.input)
        local total_sum = 0
        for i=1,#input_relevance do
          local l = get_max(input_relevance[i])
          local m, coord = l[1], l[2]
          if node.input ~= nil then
            print('Result max', i, {m, coord}, node.input[i][coord])
          else
            print('Result max', i, {m, coord})
          end
        end
      else
        local l = get_max(input_relevance)
        local m, coord = l[1], l[2]
        print('Result max', input_relevance[coord], node.input[1][coord])
      end
    end
    ]]

    if #node.mapindex == 1 then
      -- Case 1: Select node
      if node.selectindex then
        -- Initialize the selection table
        if relevances[node.mapindex[1]] == nil then
          relevances[node.mapindex[1]] = {}
        end

        if relevances[node.mapindex[1]][node.selectindex] == nil then
          relevances[node.mapindex[1]][node.selectindex] = input_relevance
        else
          relevances[node.mapindex[1]][node.selectindex]:add(input_relevance)
        end

      -- Case 2: Not select node
      else

        if relevances[node.mapindex[1]] == nil then
          relevances[node.mapindex[1]] = input_relevance
        else
          relevances[node.mapindex[1]]:add(input_relevance)
        end

      end

    else

      -- Case 3: Table uses information from several input nodes
      for j=1,#node.mapindex do
        if relevances[node.mapindex[j]] == nil then
          relevances[node.mapindex[j]] = input_relevance[j]
        else
          relevances[node.mapindex[j]]:add(input_relevance[j])
        end
      end

    end

  end

  return relevances[gmodule.innode.data]
end

function relevance_propagate(node, R, modified)
  -- For nodes without modules (like select nodes),
  -- pass through
  if node.module == nil then return R end

  local I = node.input
  local module = node.module

  -- Unpack single-element inputs
  if #I == 1 then I = I[1] end

  -- MulTable: pass-through on non-gate inputs
  if torch.typename(module) == 'nn.CMulTable' then
    -- Identify the non-gate input node
    local input_nodes = node.mapindex
    local true_index = nil
    local ultimate_inputs = {}

    for i=1,#input_nodes do
      ultimate_inputs[i] = input_nodes[i].input[1] -- We assume this is a nonlinearity

      if torch.typename(input_nodes[i].module) ~= 'nn.Sigmoid' then
        true_index = i
        break
      end
    end

    return module:lrp(I, R, true_index, modified, ultimate_inputs)
  end

  if torch.typename(module) == 'nn.CAddTable' then
    return module:lrp(I, R)
  end

  if torch.typename(module) == 'nn.Linear' then
    return module:lrp(I, R, true)
  end

  if torch.typename(module) == 'nn.Reshape' then
    return module:lrp(I, R)
  end

  if torch.typename(module) == 'nn.SplitTable' then
    return module:lrp(I, R)
  end

  if torch.typename(module) == 'nn.LookupTable' then
    -- Batch mode, so sum over second (embedding) dimension
    return R:sum(2)
  end

  -- All other cases: pass-through
  return module:lrp(I, R)
end

function nn.Module:lrp(input, relevance)
  if self.initialized_lrp == nil then
    self.initialized_lrp = true

    self.Rin = self.gradInput
  end

  self.Rin:resizeAs(relevance):copy(relevance)

  return self.Rin
end

function nn.Reshape:lrp(input, relevance)
  if self.initialized_lrp == nil then
    self.initialized_lrp = true

    self.Rin = self.gradInput
  end

  self.Rin:resizeAs(input):copy(relevance:viewAs(input))

  return self.Rin
end

function nn.SplitTable:lrp(input, relevance)
  if self.initialized_lrp == nil then
    self.initialized_lrp = true

    self.Rin = self.gradInput
  end

  local dimension = self:_getPositiveDimension(input)

  self.Rin:resizeAs(input)
  for i=1,#relevance do
    self.Rin:select(dimension, i):copy(relevance[i])
  end

  return self.Rin
end

function fits(source, dest, copy) -- Fast in-place true sign
    dest:resizeAs(source):copy(source):sign()
    copy:resizeAs(source):copy(dest):pow(2):csub(1)
    dest:csub(copy) -- Fast in-place true sign
end

-- MulTable, AddTable, Linear mirrored as closely as possible
-- from Arras's LRP_for_lstm
local eps = 0.001

-- The logit function, which inverts the sigmoid function
function inv_sigmoid(tensor)
  -- p -> 1/p
  tensor:cinv()
  -- 1/p -> 1/p - 1
  tensor:csub(1)
  -- 1/p - 1 -> log(1/p - 1)
  tensor:log()
  -- log(1/p - 1) -> -log(1/p - 1)
  return tensor:neg()
end

-- The inverse tanh function
local inv_tanh_den = torch.CudaTensor()
function inv_tanh(tensor)
  -- Inverse tanh is given as 1/2(ln((1+x)/(1-x)))
  inv_tanh_den:resizeAs(tensor):copy(tensor):neg():add(1)
  return tensor:add(1):cdiv(inv_tanh_den):log():div(2)
end

local mul_table_sum = torch.CudaTensor()
local mul_table_sum_sign = torch.CudaTensor()
local mul_table_sum_sign_clone = torch.CudaTensor()
function nn.CMulTable:lrp(input, relevance, true_index, modified, ultimate_input)
  if self.initialized_lrp == nil then
    self.initialized_lrp = true

    self.sum = mul_table_sum
    self.sum_sign = mul_table_sum_sign
    self.sum_sign_clone = mul_table_sum_sign_clone
    self.Rin = self.gradInput
  end

  -- Method from Arras, et al. https://arxiv.org/pdf/1706.07206.pdf
  if modified == nil or modified == 'Arras' then
    for i=1,#input do
      self.Rin[i] = self.Rin[i] or input[i].new()
      if i == true_index then
        self.Rin[i]:resizeAs(relevance):copy(relevance)
      else
        self.Rin[i]:resizeAs(relevance):zero()
      end
    end

    for i=#input+1,#self.Rin do
      self.Rin[i] = nil
    end

  -- Method from Ding, et al. http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_dyz.pdf
  elseif modified == 'Ding' then
    -- Get scalars and sum of scalars
    self.sum:resizeAs(relevance):zero()

    for i=1,#input do
      self.Rin[i] = self.Rin[i] or input[i].new()

      if i == true_index then
        self.Rin[i]:resizeAs(input[i]):copy(input[i]):abs()
      else
        self.Rin[i]:resizeAs(input[i]):copy(ultimate_input[i]):abs()
        --inv_sigmoid(self.Rin[i])
      end

      self.sum:add(self.Rin[i])
    end

    -- Numerical stabilizer epsilon
    fits(self.sum, self.sum_sign, self.sum_sign_clone)
    self.sum:add(self.sum_sign:mul(eps)):cinv():cmul(relevance)

    -- Normalize
    for i=1,#input do
      self.Rin[i]:cmul(self.sum)
    end

  -- New proposed method
  elseif modified == 'Proposed' then
    -- Get scalars and sum of scalars
    self.sum:resizeAs(relevance):zero()

    for i=1,#input do
      self.Rin[i] = self.Rin[i] or input[i].new()

      if i == true_index then
        self.Rin[i]:resizeAs(input[i]):copy(input[i]):div(2):abs()
      else
        self.Rin[i]:resizeAs(input[true_index]):copy(input[true_index]):div(2):csub(self.output):abs()
      end

      self.sum:add(self.Rin[i])
    end

    -- Numerical stabilizer epsilon
    fits(self.sum, self.sum_sign, self.sum_sign_clone)
    self.sum:add(self.sum_sign:mul(eps)):cinv():cmul(relevance)

    -- Normalize
    for i=1,#input do
      self.Rin[i]:cmul(self.sum)
    end
  end

  return self.Rin
end

local denom = torch.CudaTensor()
local denom_sign = torch.CudaTensor()
local denom_sign_clone = torch.CudaTensor()

function nn.Linear:lrp(input, relevance, use_bias)
  -- Allocate memory we need for LRP here.
  if self.initialized_lrp == nil then
    self.initialized_lrp = true

    self.D = self.weight:size(2)
    self.M = self.weight:size(1)

    self.denom = denom

    self.denom_sign = denom_sign
    self.denom_sign_clone = denom_sign_clone

    self.Rin = self.gradInput --torch.CudaTensor()
  end

  local b = relevance:size(1)

  -- Perform LRP propagation
  -- First, determine sign.
  self.denom:resizeAs(self.output):copy(self.output)

  fits(self.denom, self.denom_sign, self.denom_sign_clone)

  -- Add epsilon to the denominator and invert
  self.denom:add(self.denom_sign:mul(eps)):cinv():cmul(relevance)

  -- Compute main 'messages'
  self.Rin:resizeAs(input):zero()
  self.Rin:addmm(0, self.Rin, 1, self.denom, self.weight)
  self.Rin:cmul(input)

  -- Add numerator stabilizer
  self.Rin:add(self.denom_sign:cmul(self.denom):sum(2):div(self.D):view(b, 1):expandAs(self.Rin))

  -- Add bias term if present and desired
  if use_bias and self.bias then
    self.Rin:add(self.denom:cmul(self.bias:view(1, self.M):expandAs(self.denom)):sum(2):div(self.D):view(b, 1):expandAs(self.Rin))
  end

  -- Return
  return self.Rin
end

local sum_inputs = torch.CudaTensor()
local sign_sum = torch.CudaTensor()
local sign_sum_clone = torch.CudaTensor()

function nn.CAddTable:lrp(input, relevance)
  if self.initialized_lrp == nil then
    self.initialized_lrp = true

    self.sum_inputs = sum_inputs
    self.sign_sum = sign_sum
    self.sign_sum_clone = sign_sum_clone

    self.results = self.gradInput
  end

  -- Get output and stabilize
  self.sum_inputs:resizeAs(self.output):copy(self.output)
  self.sign_sum:resizeAs(self.output):copy(self.sum_inputs):sign()
  self.sign_sum_clone:resizeAs(self.output):copy(self.sign_sum):abs():csub(1)
  self.sign_sum:csub(self.sign_sum_clone) -- Fast in-place true sign

  self.sum_inputs:add(self.sign_sum:mul(eps)):cinv()

  self.sign_sum:div(#input)

  -- Scale relevance as input contributions
  for i=1,#input do
    self.results[i] = self.results[i] or input[1].new()
    self.results[i]:resizeAs(input[i]):copy(input[i]):add(self.sign_sum):cmul(self.sum_inputs):cmul(relevance)
  end

  -- Return
  for i=#input+1,#self.results do
    self.results[i] = nil
  end

  return self.results
end
