require 'cutorch'

torch.manualSeed(0)
cutorch.manualSeed(0)

local X_mean = torch.CudaTensor()
local X_stdev = torch.CudaTensor()

local Y_mean = torch.CudaTensor()
local Y_stdev = torch.CudaTensor()

local c = torch.CudaTensor()
local cabs = torch.CudaTensor()
local csign = torch.CudaTensor()
local Ga = torch.CudaTensor()
local Ga_inv = torch.CudaTensor()
local ones = torch.CudaTensor()
local w = torch.CudaTensor()
local W = torch.CudaTensor()
local Wabs = torch.CudaTensor()
local u = torch.CudaTensor()
local a = torch.CudaTensor()
local total_dist = torch.CudaTensor()
local dist_positive_denominator = torch.CudaTensor()
local dist_negative_denominator = torch.CudaTensor()
local dist_mask = torch.CudaByteTensor()
local active_mask = torch.CudaByteTensor()
local Xa = torch.CudaTensor()
local current_best_model = torch.CudaTensor()
local current_predictions = torch.CudaTensor()

local X = torch.CudaTensor()
local Y = torch.CudaTensor()
local X_valid = torch.CudaTensor()
local Y_valid = torch.CudaTensor()

local bias_term = torch.CudaTensor()

-- "Closeness" values
local lambda = 1 - 1e-10
local epsilon = 1e-2

function lars(X_in, Y_in, X_valid_in, Y_valid_in, lasso_path)
  X:resizeAs(X_in):copy(X_in)
  Y:resizeAs(Y_in):copy(Y_in)

  local samples = X:size(1)
  local inputs = X:size(2)
  local recovering = false

  -- Normalize X
  X_mean:resize(inputs):mean(X, 1)
  X:csub(X_mean:view(1, inputs):expandAs(X))
  X_stdev:resize(inputs):std(X, 1)
  X:cdiv(X_stdev:view(1, inputs):expandAs(X))

  -- Normalize Y:
  Y_mean:resize(1):mean(Y, 1)
  Y:csub(Y_mean:resize(1, 1):expandAs(Y))

  -- Normalize validation
  X_valid:resizeAs(X_valid_in):copy(X_valid_in)
  Y_valid:resizeAs(Y_valid_in):copy(Y_valid_in)

  -- (X)
  X_mean:resize(inputs):mean(X_valid, 1)
  X_valid:csub(X_mean:view(1, inputs):expandAs(X_valid))
  X_stdev:resize(inputs):std(X_valid, 1)
  X_valid:cdiv(X_stdev:view(1, inputs):expandAs(X_valid))

  -- (Y)
  Y_mean:resize(1):mean(Y_valid, 1)
  Y_valid:csub(Y_mean:resize(1, 1):expandAs(Y_valid))

  -- Get covariances; this will have size (inputs) x 1
  c:resize(inputs, 1):zero()
  c:addmm(0, c, 1, X:t(), Y)

  cabs:resizeAs(c)
  csign:resizeAs(c)

  total_dist:resize(2 * inputs, 1)
  local dist_positive = total_dist:narrow(1, 1, inputs)
  local dist_negative = total_dist:narrow(1, inputs+1, inputs)

  dist_positive_denominator:resizeAs(dist_positive)
  dist_negative_denominator:resizeAs(dist_negative)

  active_mask:resize(inputs):zero()
  local active = {}

  -- Weight matrix
  W:resize(inputs, 1):zero()
  Wabs:resizeAs(W):zero()

  -- Cross-validation state
  current_best_model:resizeAs(W):zero()
  current_predictions:resizeAs(Y_valid):zero()
  local current_best_error = math.huge

  local just_removed = false
  cabs:copy(c):abs()

  local cmax = cabs:max()
  for i=1,inputs do
    if cabs[i][1] > cmax * lambda - epsilon then
      table.insert(active, i)
      active_mask[i] = 1
    end
  end

  while cabs:sum() > epsilon * inputs do
    csign:copy(c):sign()

    -- Select the active columns of the inputs matrix
    Xa:resize(samples, #active)
    for i=1,#active do
      Xa:narrow(2, i, 1):copy(
        X:narrow(2, active[i], 1)
      ):mul(csign[active[i]][1])
    end

    -- Compute the equiangular vector.
    --
    -- TODO this could be sped up by a linear factor with torch.gesv, unsure
    -- whether this would be faster or slower than CUDA inverse.
    Ga:resize(#active, #active):zero()
    Ga:addmm(0, Ga, 1, Xa:t(), Xa)

    Ga_inv:inverse(Ga)

    ones:resize(#active):fill(1)
    w:resize(#active):zero()
    w:addmv(0, w, 1, Ga_inv, ones)

    local A = 1 / math.sqrt(w:sum())
    u:resize(samples)
    u:addmv(0, u, A, Xa, w)

    a:resize(inputs)
    a:addmv(0, a, 1, X:t(), u)

    -- Compute distances to equal correlations
    dist_positive:copy(c):add(cmax)
    dist_positive_denominator:copy(a):add(A)
    dist_positive:cdiv(dist_positive_denominator)

    dist_negative:copy(c):csub(cmax)
    dist_negative_denominator:copy(a):csub(A)
    dist_negative:cdiv(dist_negative_denominator)

    local gamma
    if #active == inputs then
      gamma = cmax / A
    else
      gamma = math.huge
      for i=1,inputs do
        if recovering or active_mask[i] == 0 then
          if dist_positive[i][1] > 0 and dist_positive[i][1] < gamma then
            gamma = dist_positive[i][1]
          end
          if dist_negative[i][1] > 0 and dist_negative[i][1] < gamma then
            gamma = dist_negative[i][1]
          end
        end
      end

      recovering = false

      if gamma == math.huge then
        print('Breaking due to some error.')
        break
      end

      -- Lasso path modification, if specified
      local subgamma = math.huge

      inactive_indices = {}
      if lasso_path then
        for i=1,#active do
          local weight = W[active[i]][1]
          local delta = csign[active[i]][1] * w[i] * A
          local candidate_subgamma = -weight / delta

          if candidate_subgamma > 0 and candidate_subgamma < subgamma then
            subgamma = candidate_subgamma
            inactive_indices[active[i]] = true
          elseif candidate_subgamma > 0 and candidate_subgamma < subgamma - epsilon then
            inactive_indices[active[i]] = true
          end
        end
      end

      if subgamma < gamma then
        gamma = subgamma
        recovering = true
      end
    end

    -- Increase parameters by w * gamma
    for i=1,#active do
      W[active[i]]:add(w[i] * csign[active[i]][1] * A * gamma)
    end

    -- Compute new model error for stopping criterion
    current_predictions:addmm(0, current_predictions, 1, X_valid, W)
    local new_error = math.sqrt(current_predictions:csub(Y_valid):pow(2):mean())
    if new_error < current_best_error then
      --print('A model with', #active, 'inputs performs better.')
      current_best_model:copy(W)
      current_best_error = new_error
    end

    -- Decrease correlations by a * gamma
    c:csub(a:mul(gamma))
    cmax = cmax - A * gamma

    -- Recompute actives
    active = {}
    active_mask:zero()

    cabs:copy(c):abs()
    cmax = cabs:max()
    for i=1,inputs do
      if cabs[i][1] > cmax * lambda - epsilon then
        -- During lasso path computation, exclude things that are crossing zero
        if not inactive_indices[i] then
          table.insert(active, i)
          active_mask[i] = 1
        end
      end
    end
  end

  return current_best_model --W:cdiv(X_stdev:view(inputs, 1):expandAs(W))
end

-- TESTING
--[[

-- Train
local X_true = torch.CudaTensor(1000, 50):normal()
local W_true = torch.CudaTensor(50, 1):zero()
local variance = 40

for i=1,10 do
  W_true[i]:normal()
end

local Y_true = torch.CudaTensor(1000, 1):uniform()
Y_true:addmm(variance, Y_true, 1, X_true, W_true)


-- Validation
local X_validation = torch.CudaTensor(500, 50):normal()
local Y_validation = torch.CudaTensor(500, 1):uniform() * variance + X_validation * W_true

local ols = torch.inverse(X_true:t() * X_true) * X_true:t() * Y_true
local lars_soln = lars(X_true, Y_true, X_validation, Y_validation, true)

print('PREDICTORS: (true, ols, lasso)')
print(torch.cat(
  {W_true, ols, lars_soln}
))

-- Test
local other_X = torch.CudaTensor(1000, 50):normal()
local other_Y = torch.CudaTensor(1000, 1):uniform() * variance + other_X * W_true
print('ERRORS (as unaccounted variance):')
print(
    math.sqrt((other_X * ols - other_Y):pow(2):mean()),
    math.sqrt((other_X * lars_soln - other_Y):pow(2):mean())
)

--[[
print('')
print(lars(X_true, Y_true, math.huge, false, true))
]]
