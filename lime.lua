require 'lars'

-- Importance measured by local linear approximation
-- a simplified version of LIME: https://arxiv.org/pdf/1602.04938v1.pdf

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

local source = torch.CudaTensor()
local input_matrix = torch.CudaTensor()
local output_matrix = torch.CudaTensor()
local input_validation = torch.CudaTensor()
local output_validation = torch.CudaTensor()
local first_hidden = {}

local base = torch.CudaTensor()
local modifier = torch.CudaTensor()
local inverse_modifier = torch.CudaTensor()
local projection = torch.CudaTensor()

-- Softmax layer
local softmax = nn.Sequential()
softmax:add(nn.SoftMax())
softmax:cuda()

function lime(
    alphabet,
    model,
    normalizer,
    sentence,
    num_perturbations,
    perturbation_size)

  local opt, encoder_clones, linear = model.opt, model.clones, model.linear

  local mean, stdev = normalizer[1], normalizer[2]

  source:resize(opt.rnn_size, #alphabet)

  -- Construct beginning hidden state
  for i=1,2*opt.num_layers do
    first_hidden[i] = first_hidden[i] or torch.CudaTensor()
    first_hidden[i]:resize(opt.rnn_size, opt.rnn_size)
  end

  -- Gradient-retrieval function
  function run_forward(input_vector, output_vector)
    -- Forward pass
    local rnn_state = first_hidden
    local perturbed_encodings
    for t=1,#sentence do
      source:uniform()
      source:narrow(2, sentence[t], 1):mul(perturbation_size * 2)

      local softmaxed = softmax:forward(source)

      local encoder_input = {
        linear:forward(
          softmax:forward(source)
        )
      }

      -- Record amount of perturbation
      input_vector:narrow(2, t, 1):copy(
        softmaxed:narrow(2, sentence[t], 1)
      )

      append_table(encoder_input, rnn_state)
      rnn_state = encoder_clones[t]:forward(encoder_input)
    end

    -- Compute and record normalized loss
    output_vector:copy(rnn_state[#rnn_state]):csub(
      mean:expandAs(rnn_state[#rnn_state])
    ):cdiv(
      stdev:expandAs(rnn_state[#rnn_state])
    )

    return perturbed_encodings, loss
  end

  local lime_data_inputs = {}
  local lime_data_outputs = {}

  -- Do several perturbations
  input_matrix:resize(num_perturbations * opt.rnn_size, #sentence)
  output_matrix:resize(num_perturbations * opt.rnn_size, opt.rnn_size)

  for i=1,num_perturbations do
    -- Create the data point for LIME to regress from
    run_forward(
      input_matrix:narrow(1, (i-1)*opt.rnn_size+1, opt.rnn_size),
      output_matrix:narrow(1, (i-1)*opt.rnn_size+1, opt.rnn_size)
    )
  end

  -- Also for validation
  input_validation:resize(num_perturbations * opt.rnn_size, #sentence)
  output_validation:resize(num_perturbations * opt.rnn_size, opt.rnn_size)

  for i=1,num_perturbations do
    -- Create the data point for LIME to regress from
    run_forward(
      input_validation:narrow(1, (i-1)*opt.rnn_size+1, opt.rnn_size),
      output_validation:narrow(1, (i-1)*opt.rnn_size+1, opt.rnn_size)
    )
  end

  -- Create the local linear model
  -- Projection should be length x 1

  projection:resize(#sentence, opt.rnn_size):zero()
  -- Result projection
  --[[
  -- X:t() * Y
  base:resize(#sentence, opt.rnn_size):zero()
  -- X:t() * X
  modifier:resize(#sentence, #sentence):zero()

  -- Matrix multiply
  modifier:addmm(0, modifier, 1, input_matrix:t(), input_matrix)
  -- Invert
  inverse_modifier:resize(#sentence, #sentence)
  torch.inverse(inverse_modifier, modifier)

  -- Matrix multiply
  base:addmm(0, base, 1, input_matrix:t(), output_matrix)
  -- Matrix multiply
  projection:addmm(0, projection, 1, inverse_modifier, base)
  ]]

  -- Perform LASSO regression with cross-validation.
  for i=1,opt.rnn_size do
    projection:narrow(2, i, 1):copy(
      lars(
        input_matrix,
        output_matrix:narrow(2, i, 1),
        input_validation,
        output_validation:narrow(2, i, 1),
        true
      )
    )
  end

  -- Get affinity for each token in the sentence
  local affinity = {}
  for t=1,#sentence do
    table.insert(affinity, projection[t])
  end

  return affinity
end
