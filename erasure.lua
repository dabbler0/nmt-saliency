-- Importance measured by word erasure:
-- implemented from https://arxiv.org/pdf/1612.08220.pdf

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

local results = torch.CudaTensor()
local first_hidden = {}

function erasure(
    alphabet,
    model,
    normalizer,
    sentence)

  local opt, encoder_clones, lookup = model.opt, model.clones, model.lookup

  local mean, stdev = normalizer[1], normalizer[2]

  -- Construct beginning hidden state
  for i=1,2*opt.num_layers do
    first_hidden[i] = first_hidden[i] or torch.CudaTensor()
    first_hidden[i]:resize(1, opt.rnn_size):zero()
  end
  for i=2*opt.num_layers+1,#first_hidden do
    first_hidden[i] = nil
  end

  -- Gradient-retrieval function
  function run_forward(skip, output, limit)
    -- Forward pass
    local rnn_state = first_hidden
    for t=1,limit do
      -- Skip the given token
      if t >= skip then t = t + 1 end

      local encoder_input = {lookup:forward(torch.CudaTensor{sentence[t]})}
      append_table(encoder_input, rnn_state)
      rnn_state = encoder_clones[1]:forward(encoder_input)
    end

    -- Return entire output vector
    -- as a 1d tensor
    output:copy(rnn_state[#rnn_state][1]):csub(mean[1]):cdiv(stdev[1])
  end

  -- Do several perturbations
  local length = #sentence
  results:resize(#sentence+1, opt.rnn_size)
  for t=1,#sentence do
    run_forward(t, results[t], #sentence-1)
  end
  run_forward(#sentence+1, results[#sentence+1], #sentence)

  local reference = results[#sentence+1]

  -- Get affinity for each token in the sentence
  local affinity = {}
  for t=1,#sentence do
    table.insert(affinity, results[t]:csub(reference):neg())
  end

  return affinity
end
