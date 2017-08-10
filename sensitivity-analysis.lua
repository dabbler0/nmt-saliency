-- Importance as measured by gradients at slightly permuted positions, as described here:
-- https://arxiv.org/pdf/1706.03825.pdf
function printMemProfile()
  free, total = cutorch.getMemoryUsage()
  print('GPU MEMORY FREE: ', free, 'of', total)
end

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
local sentence_source = torch.CudaLongTensor()
local first_hidden = {}
local last_hidden = {}

function sensitivity_analysis(
    alphabet,
    model,
    normalizer,
    sentence)

  local opt, encoder_clones, lookup_table = model.opt, model.clones, model.lookup

  local alphabet_size = #alphabet
  local length = #sentence

  local mean, stdev = normalizer[1], normalizer[2]

  affinities:resize(length, opt.rnn_size):zero()

  local source_gradients = {}

  -- Construct beginning hidden state
  for i=1,2*opt.num_layers do
    first_hidden[i] = first_hidden[i] or torch.CudaTensor()
    first_hidden[i]:resize(opt.rnn_size, opt.rnn_size):zero()
  end

  print('FIRST CHECKPOINT')
  printMemProfile()

  -- Forward pass
  local rnn_state = first_hidden
  local encoder_inputs = {}
  for t=1,length do
    local encoder_input = {
      lookup_table:forward(
        sentence_source:resize(1):fill(sentence[t]):expand(opt.rnn_size)
      )
    }
    append_table(encoder_input, rnn_state)
    encoder_inputs[t] = encoder_input
    rnn_state = encoder_clones[t]:forward(encoder_input)
  end

  print('SECOND CHECKPOINT')
  printMemProfile()

  -- Backward pass

  -- Construct final gradient
  for i=1,2*opt.num_layers do
    last_hidden[i] = last_hidden[i] or torch.CudaTensor()
    last_hidden[i]:resize(opt.rnn_size, opt.rnn_size):zero()
  end

  -- Diagonal matrix of normalizers. This represents 500 batches, one for each dimensions,
  -- where dimension x is backpropagating relevance for the xth output
  last_hidden[#last_hidden]:diag(stdev[1]:cinv())

  -- We just inverted stdev[1] so undo that
  stdev[1]:cinv()

  print('THIRD CHECKPOINT')
  printMemProfile()

  -- Initialize.
  local rnn_state_gradients = {}
  rnn_state_gradients[length] = last_hidden

  for t=length,1,-1 do
    local encoder_input_gradient = encoder_clones[t]:backward(encoder_inputs[t], rnn_state_gradients[t])
    -- Get source gradients and copy into gradient array
    local final_gradient = encoder_input_gradient[1]

    affinities[t]:sum(final_gradient:pow(2), 2)

    -- Get RNN state gradients
    rnn_state_gradients[t-1] = slice_table(encoder_input_gradient, 2)
  end

  -- Get affinity for each token in the sentence
  local result_affinity = {}
  for t=1,length do
    table.insert(result_affinity, affinities[t])
  end

  return result_affinity
end

