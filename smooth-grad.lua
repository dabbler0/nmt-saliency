-- Importance as measured by gradients at slightly permuted positions, as described here:
-- https://arxiv.org/pdf/1706.03825.pdf
function append_table(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

function slice_table(t, index)
  result = {}
  for i=index,#t do
    table.insert(result, t[i])
  end
  return result
end

local affinities = torch.CudaTensor()
local affinities_clone = torch.CudaTensor()
local cumulative_affinities = torch.CudaTensor()
local source = torch.CudaTensor()
local first_hidden = {}
local last_hidden = {}

-- Softmax layer
local softmax = nn.Sequential()
softmax:add(nn.SoftMax())
softmax:cuda()

function smooth_grad(
    alphabet,
    model,
    normalizer,
    sentence,
    num_perturbations,
    perturbation_size)

  affinities:resize(#sentence, opt.rnn_size):zero()
  affinities_clone:resizeAs(affinities):zero()
  cumulative_affinities:resizeAs(affinities):zero()

  source:resize(#alphabet):zero()

  -- Default arguments
  if num_perturbations == nil then num_perturbations = 3 end
  if perturbation_size == nil then perturbation_size = 11 end

  local opt, encoder_clones, linear = model.opt, model.clones, model.linear

  local mean, stdev = normalizer[1], normalizer[2]

  -- Construct beginning hidden state
  for i=1,2*opt.num_layers do
    first_hidden[i] = first_hidden[i] or torch.CudaTensor()
    first_hidden[i]:resize(opt.rnn_size, opt.rnn_size):zero()
  end
  for i=2*opt.num_layers+1,#first_hidden do
    first_hidden[i] = nil
  end

  -- Gradient-retrieval function
  function get_gradient()
    affinities:zero()

    -- Forward pass
    local rnn_state = first_hidden
    local softmax_derivatives = {}
    for t=1,#sentence do
      source:uniform()
      source[sentence[t]] = perturbation_size

      local softmaxed = softmax:forward(source)

      local encoder_input = {
        linear:forward(
          softmaxed
        ):view(1, opt.rnn_size):expand(opt.rnn_size, opt.rnn_size)
      }
      append_table(encoder_input, rnn_state)

      table.insert(softmax_derivatives, softmaxed[sentence[t]] * (1 - softmaxed[sentence[t]]))

      rnn_state = encoder_clones[t]:forward(encoder_input)
    end

    -- Compute normalized loss
    local loss = (rnn_state[#rnn_state][1] - mean[1]):cdiv(stdev[1])

    -- Backward pass

    -- Construct final gradient
    for i=1,2*opt.num_layers do
      last_hidden[i] = last_hidden[i] or torch.CudaTensor()
      last_hidden[i]:resize(opt.rnn_size, opt.rnn_size)
    end

    -- Set the last hidden to this diagonal matrix containing the derivatives of the normalized
    -- activations wrt the actual activations
    last_hidden[2 * opt.num_layers]:diag(stdev[1]:cinv())

    -- We just inverted stdev in place so undo that
    stdev[1]:cinv()

    -- Initialize.
    local rnn_state_gradients = {}
    rnn_state_gradients[#sentence] = last_hidden

    for t=#sentence,1,-1 do
      local encoder_input_gradient = encoder_clones[t]:backward(
        encoder_clones[t].innode.input, -- Use existing stored input
        rnn_state_gradients[t]
      )

      local embedding_gradient = encoder_input_gradient[1]

      -- The gradient wrt the softmaxed one-hot index Oi is
      -- df/dE * dE/dOi
      affinities[t]:addmv(
        0,
        affinities[t],
        softmax_derivatives[t], -- softmax derivative s(1 - s)
        embedding_gradient, linear.weight[{{}, sentence[t]}]
      )

      -- Get RNN state gradients
      rnn_state_gradients[t-1] = slice_table(encoder_input_gradient, 2)
    end

    return affinities
  end

  -- Do several perturbations
  for i=1,num_perturbations do
    cumulative_affinities:add(get_gradient())
  end

  -- Average
  cumulative_affinities:div(num_perturbations)

  -- Get affinity for each token in the sentence
  local result_affinity = {}
  for t=1,#sentence do
    table.insert(result_affinity, cumulative_affinities[t])
  end

  return result_affinity
end

