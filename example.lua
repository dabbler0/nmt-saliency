-- Compute saliencies by four methods:
--  - Word erasure
--  - Layer-wise Relevance Propagation
--  - SmoothGrad
--  - LIME

require 'nn'
require 'nngraph'
require 'hdf5'
require 'optim'

require 'cutorch'
require 'cunn'

require 's2sa.data'
require 's2sa.models'
require 's2sa.model_utils'

require 'smooth-grad'
require 'layerwise-relevance-propagation'
require 'sensitivity-analysis'
require 'lime'
require 'erasure'

require 'json'

torch.manualSeed(0)

function head_table(t, index)
  result = {}
  for i=1,index do
    table.insert(result, t[i])
  end
  return result
end

-- Split the first layer (word embedding layer) off from the
-- rest of the network, so that it can be run differently.
function create_split_encoder_clones(checkpoint, max_len)
  local encoder, opt = checkpoint[1][1], checkpoint[2]

  local lookuptable_module = nil
  local linear_module = nil

  encoder:replace(function(module)
    if torch.typename(module) == 'nn.LookupTable' then
      local weight = module.weight
      local layer = nn.Linear(weight:size(1), weight:size(2), false)

      layer.weight = weight:t()

      linear_module = layer
      lookuptable_module = module

      return nn.Identity()
    else
      return module
    end
  end)

  encoder:evaluate()
  encoder = encoder:cuda()
  lookuptable_module = lookuptable_module:cuda()
  linear_module = linear_module:cuda()

  return {
    ['lookup'] = lookuptable_module,
    ['linear'] = linear_module,
    ['clones'] = clone_many_times(encoder, max_len),
    ['opt'] = opt
  }
end

function idx2key(file)
  local f = io.open(file,'r')
  local t = {}
  for line in f:lines() do
    local c = {}
    for w in line:gmatch'([^%s]+)' do
      table.insert(c, w)
    end
    t[tonumber(c[2])] = c[1]
  end
  return t
end

function invert_table(t)
  r = {}
  for k, v in ipairs(t) do
    r[v] = k
  end
  return r
end

function token_length(line)
  local k = 1
  for entry in line:gmatch'([^%s]+)' do
    k = k + 1
  end
  return k
end

function tokenize(line, inverse_alphabet)
  -- Tokenize the start line
  local tokens = {
    inverse_alphabet['<s>']
  }
  local k = 0
  for entry in line:gmatch'([^%s]+)' do
    table.insert(tokens,
      inverse_alphabet[entry] or inverse_alphabet['<unk>']
    )
  end

  return tokens
end

function deep_totable(tbl)
  local result = {}
  for i=1,#tbl do
    table.insert(result, tbl[i]:totable())
  end
  return result
end

-- Get all saliencies from all methods.
-- This can be modified to exclude some methods.
function get_all_saliencies(
    alphabet,
    model,
    normalizer,
    sentence,
    num_perturbations,
    perturbation_size)

  -- Find raw activations.
  opt = model['opt']
  encoder_clones = model['clones']
  lookup_layer = model['lookup']

  local start = os.clock()

  -- rnn state
  local rnn_state = {}
  for i=1,2*opt.num_layers do
    table.insert(rnn_state, torch.Tensor(1, opt.rnn_size):zero():cuda())
  end

  local activations = {}
  for t=1,#sentence do
    local inp = {lookup_layer:forward(torch.CudaTensor{sentence[t]})}
    append_table(inp, rnn_state)
    rnn_state = encoder_clones[t]:forward(inp)
    activations[t] = (rnn_state[#rnn_state][1] - normalizer[1][1]):cdiv(normalizer[2][1])
  end
  local act_elapsed_time = os.clock()
  print('act elapsed:', act_elapsed_time - start)
  start = os.clock()

  -- First-derivative sensitivity analysis
  local sa = sensitivity_analysis(
      alphabet,
      model,
      normalizer,
      sentence
  )
  local sa_elapsed_time = os.clock() - start
  print('sa elapsed:', sa_elapsed_time)
  start = os.clock()

  -- SmoothGrad saliency
  local smooth_grad_saliency = smooth_grad(
      alphabet,
      model,
      normalizer,
      sentence,
      num_perturbations,
      perturbation_size
  )
  local sgrad_elapsed_time = os.clock() - start
  print('sgrad elapsed:', sgrad_elapsed_time)
  start = os.clock()

  -- LRP saliency
  local layerwise_relevance_saliency = LRP_saliency(
      alphabet,
      model,
      normalizer,
      sentence
  )
  local lrp_elapsed_time = os.clock() - start
  print('lrp elapsed:', lrp_elapsed_time)
  start = os.clock()

  local lime_saliency = lime(
      alphabet,
      model,
      normalizer,
      sentence,
      2, -- 1000 data points for lime should be more than enough.
      perturbation_size
  )

  local lime_elapsed_time = os.clock() - start
  print('lime elapsed:', lime_elapsed_time)
  start = os.clock()

  local erasure_saliency = erasure(
      alphabet,
      model,
      normalizer,
      sentence
  )
  local erasure_elapsed_time = os.clock() - start
  print('erasure elapsed:', erasure_elapsed_time)

  return {
    ['saliencies'] = {
      ['sgrad'] = deep_totable(smooth_grad_saliency),
      ['lrp'] = deep_totable(layerwise_relevance_saliency),
      ['lime'] = deep_totable(lime_saliency),
      ['erasure'] = deep_totable(erasure_saliency),
      ['sa'] = deep_totable(sa),
    },
    ['times'] = {
      ['sgrad'] = sgrad_elapsed_time,
      ['lrp'] = lrp_elapsed_time,
      ['lime'] = lime_elapsed_time,
      ['erasure'] = erasure_elapsed_time,
      ['sa'] = sa_elapsed_time
    },
    ['activations'] = activations
  }

end

function main()
  cmd = torch.CmdLine()

  cmd:option('-model', '', '_final.t7 model location')
  cmd:option('-dict', '', 'src.dict location')
  cmd:option('-description', '', 'description (OPTIONAL) -- a .t7 file of sampled activations of neurons, used to norm stdev to 1 and mean to 0. Generated by describe.lua from seq2seq-comparison.')
  cmd:option('-max_len', 30, 'Maximum length (initialization time + memory usage most affected by this)')
  cmd:option('-num_perturbations', 3, 'Number of perturbations for SmoothGrad to use. 3 is usually sufficient.')
  cmd:option('-perturbation_size', 11, 'Larger number means smaller perturbation.')

  local opt = cmd:parse(arg)

  local models = {}
  local normalizers = {}

  -- Model file
  local model = create_split_encoder_clones(torch.load(opt.model), opt.max_len)

  -- Alphabet file
  local alphabet = idx2key(opt.dict)
  local inverse_alphabet = invert_table(alphabet)

  -- Normalizer; if an activation distribution is provided, norm to 
  local normalizer
  if opt.description ~= '' then
    local encodings = torch.load(opt.description)['encodings']
    local concatenated = torch.cat(encodings, 1):cuda()

    local mean = concatenated:mean(1)
    concatenated:csub(mean:view(1, concatenated:size(2)):expandAs(concatenated))

    local stdev = concatenated:pow(2):mean(1):sqrt()
    normalizer = {mean, stdev}

  -- If one is not, just do no normalization.
  else
    normalizer = {torch.zeros(1, model['opt'].rnn_size), torch.ones(1, model['opt'].rnn_size)}
  end

  io.stderr:write('Loaded and prepared model.\n')

  -- EXAMPLE USAGE
  -- (the code below is what should probably be changed to create frameworks for things)

  while true do
    -- Select a sentence to run saliency over
    local sentence = tokenize(io.read(), inverse_alphabet)

    print('Getting all saliencies.')

    -- Get all saliencies. Note that this gets it for every neuron, but we don't use them all.
    local saliencies = get_all_saliencies(
      alphabet,
      model,
      normalizer,
      sentence,
      opt.num_perturbations,
      opt.perturbation_size
    )

    -- Select a neuron to inspect
    local neuron = tonumber(io.read())


    -- Print out the activations for this neuron first
    local str = 'activat'
    local p = saliencies['activations']
    for i=1,#p do
      str = str .. '\t' .. string.format('%.3f', p[i][neuron])
    end
    print(str)

    -- Then print out each input word's saliency according to saliency metrics
    for k,p in pairs(saliencies['saliencies']) do
      str = k
      for i=1,#p do
        str = str .. '\t' .. string.format('%.3f', p[i][neuron])
      end
      print(str)
    end
  end

  -- END EXAMPLE USAGE
end

main()
