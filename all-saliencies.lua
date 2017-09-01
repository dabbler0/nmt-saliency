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

function printMemProfile()
  free, total = cutorch.getMemoryUsage()
  print('GPU MEMORY FREE: ', free, 'of', total)
end

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

  -- Don't insert the end-of-sentence token,
  -- as we're trying to examine the activation right here

  --table.insert(tokens, inverse_alphabet['</s>'])

  return tokens
end

function deep_totable(tbl)
  local result = {}
  for i=1,#tbl do
    table.insert(result, tbl[i]:totable())
  end
  return result
end

local rnn_state = {}
local activations = torch.CudaTensor()
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
  for i=1,2*opt.num_layers do
    rnn_state[i] = rnn_state[i] or torch.CudaTensor()
    rnn_state[i]:resize(1, opt.rnn_size):zero()
  end

  activations:resize(#sentence, opt.rnn_size)
  for t=1,#sentence do
    local inp = {lookup_layer:forward(torch.CudaTensor{sentence[t]})}
    append_table(inp, rnn_state)
    rnn_state = encoder_clones[t]:forward(inp)
    activations[t]:copy(rnn_state[#rnn_state][1]):csub(normalizer[1][1]):cdiv(normalizer[2][1])
  end
  local act_elapsed_time = os.clock()
  print('act elapsed:', act_elapsed_time - start)
  start = os.clock()

  printMemProfile()

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

  printMemProfile()

  -- LRP saliency
  local layerwise_relevance_saliency = deep_totable(LRP_saliency(
      alphabet,
      model,
      normalizer,
      sentence,
      'Arras'
  ))
  local lrp_elapsed_time = os.clock() - start
  print('Arras lrp elapsed:', lrp_elapsed_time)
  start = os.clock()

  printMemProfile()

  local erasure_saliency = erasure(
      alphabet,
      model,
      normalizer,
      sentence
  )
  local erasure_elapsed_time = os.clock() - start
  print('erasure elapsed:', erasure_elapsed_time)

  printMemProfile()

  return {
    ['saliencies'] = {
      ['sgrad'] = deep_totable(smooth_grad_saliency),
      ['lrp'] = (layerwise_relevance_saliency),
      ['erasure'] = deep_totable(erasure_saliency),
    },
    ['times'] = {
      ['sgrad'] = sgrad_elapsed_time,
      ['lrp'] = lrp_elapsed_time,
      ['erasure'] = erasure_elapsed_time,
    },
    ['activations'] = activations:totable()
  }

end

function main()
  cmd = torch.CmdLine()

  cmd:option('-model_list', '', 'List of models with alphabets (alternating lines)')
  cmd:option('-src_file', '', 'Source (en.tok) file for sampling over')
  cmd:option('-out_file', '', 'Output file to write json descriptions')
  cmd:option('-max_len', 30, 'Maximum length')
  cmd:option('-num_perturbations', 3, 'Perturbations scale; LIME will use 10n while SmoothGrad will use n')
  cmd:option('-perturbation_size', 11, 'Larger number means smaller perturbation')

  local opt = cmd:parse(arg)

  local models = {}
  local alphabets = {}
  local inverse_alphabets = {}
  local normalizers = {}

  io.stderr:write('Opening model list ' .. opt.model_list .. '\n')
  local model_file = io.open(opt.model_list)
  while true do
    local model_name = model_file:read("*line")
    if model_name == nil then break end
    local model_key = model_name:match("%a%a%-%a%a%-%d")
    io.stderr:write('Loading ' .. model_name .. ' as ' .. model_key .. '\n')
    models[model_key] = create_split_encoder_clones(torch.load(model_name), opt.max_len)

    local dict_name = model_file:read("*line")
    if dict_name == nil then break end
    alphabets[model_key] = idx2key(dict_name)
    inverse_alphabets[model_key] = invert_table(alphabets[model_key])

    io.stderr:write('Computing normalizer...\n')
    local desc_name = model_file:read("*line")
    if desc_name == nil then break end

    -- Collect encodings
    local encodings = torch.load(desc_name)['encodings']
    local concatenated = torch.cat(encodings, 1):cuda()

    -- Get mean
    local mean = concatenated:mean(1)
    concatenated:csub(mean:view(1, concatenated:size(2)):expandAs(concatenated))

    -- Get stdev
    local stdev = concatenated:pow(2):mean(1):sqrt()
    normalizers[model_key] = {mean, stdev}
  end
  model_file:close()

  io.stderr:write('Loaded all models.\n')

  local sample_file = io.open(opt.src_file)

  local line_no = 1
  local indices = {}

  local out_file = io.open(opt.out_file, 'w')

  while true do
    -- Read line
    local line = sample_file:read("*line")
    if line == nil then break end

    local net_description = {}
    local len = token_length(line)
    if opt.max_len > len then
      indices[line_no] = torch.random(1, len)
    else
      indices[line_no] = torch.random(1, opt.max_len)
    end

    print(line_no, 'PROCESSING LINE', line, 'AT INDEX', indices[line_no])

    -- Iterate over models
    for model_name, model in pairs(models) do
      -- Get saliency for this model.

      -- First, tokenize with this model's tokenizer
      local tokens = tokenize(line, inverse_alphabets[model_name])

      -- Slice off tokens at a chosen random index
      tokens = head_table(tokens, indices[line_no])

      -- (Regenerate the original non-unk tokens for visualization purposes)
      local recognized_tokens = {}
      for i=1,#tokens do
        table.insert(recognized_tokens, alphabets[model_name][tokens[i]])
      end

      -- Get saliencies
      local saliencies = get_all_saliencies(
        alphabets[model_name],
        models[model_name],
        normalizers[model_name],
        tokens,
        opt.num_perturbations,
        opt.perturbation_size
      )

      -- Store
      net_description[model_name] = saliencies
    end

    -- Write this line to a line in the output
    out_file:write(json.encode{
      ['index'] = indices[line_no],
      ['line'] = line,
      ['description'] = net_description
    })
    out_file:write('\n')
    out_file:flush()

    line_no = line_no + 1
  end
  sample_file:close()
end

main()
