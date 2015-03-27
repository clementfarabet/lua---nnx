
local nnxtest = {}
local precision = 1e-5
local mytester

-- you can easily test specific units like this: 
-- th -lnnx -e "nnx.test{'MultiSoftMax'}"
-- th -lnnx -e "nnx.test{'SoftMaxTree', 'Balance'}"

function nnxtest.SpatialPadding()
   local fanin = math.random(1,3)
   local sizex = math.random(4,16)
   local sizey = math.random(4,16)
   local pad_l = math.random(0,8)
   local pad_r = math.random(0,8)
   local pad_t = math.random(0,8)
   local pad_b = math.random(0,8)
   local val = torch.randn(1):squeeze()
   local module = nn.SpatialPadding(pad_l, pad_r, pad_t, pad_b, val)
   local input = torch.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.SpatialLinear()
   local fanin = math.random(1,10)
   local fanout = math.random(1,10)
   local in1 = torch.rand(fanin,1,1)
   local module = nn.SpatialLinear(fanin,fanout)
   local moduleg = nn.Linear(fanin,fanout)
   moduleg.weight:copy(module.weight)
   moduleg.bias:copy(module.bias)
   local out = module:forward(in1)
   local ground = moduleg:forward(in1:select(2,1,1):select(2,1,1))
   local err = out:dist(ground)
   mytester:assertlt(err, precision, torch.typename(module) .. ' - forward err ')

   local fanin = math.random(1,10)
   local fanout = math.random(1,10)
   local sizex = math.random(4,16)
   local sizey = math.random(4,16)
   local module = nn.SpatialLinear(fanin, fanout)
   local input = torch.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = nn.Jacobian.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err, precision, 'error on weight ')

   local err = nn.Jacobian.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err, precision, 'error on bias ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.SpatialMaxPooling()
   local fanin = math.random(1,4)
   local osizex = math.random(1,4)
   local osizey = math.random(1,4)
   local mx = math.random(2,6)
   local my = math.random(2,6)
   local sizex = osizex*mx
   local sizey = osizey*my
   local module = nn.SpatialMaxPooling(mx,my)
   local input = torch.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

local function template_SpatialReSamplingEx(up, mode)
   for iTest = 1,3 do
      local nDims = math.random(2,6)
      local dims = torch.LongStorage(nDims)
      for i = 1,nDims do
	 dims[i] = math.random(5,20/nDims)
      end
      local xratio, yratio
      if up then
	 xratio = torch.uniform(1.5, 10)
	 yratio = torch.uniform(1.5, 10)
      else
	 xratio = torch.uniform(0.41, 0.7)
	 yratio = torch.uniform(0.41, 0.7)
      end
      local ydim = math.random(1,nDims-1)
      local xdim = ydim+1
      local owidth_ = math.floor(dims[xdim]*xratio+0.5)
      local oheight_ = math.floor(dims[ydim]*yratio+0.5)
      local module = nn.SpatialReSamplingEx({owidth=owidth_, oheight=oheight_,
					     xDim=xdim, yDim = ydim, mode=mode})
      local input = torch.rand(dims)
      
      local err = nn.Jacobian.testJacobian(module, input)
      mytester:assertlt(err, precision, 'error on state ')
      
      local ferr, berr = nn.Jacobian.testIO(module, input)
      mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
      mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   end
end

function nnxtest.SpatialReSamplingEx1() template_SpatialReSamplingEx(true , 'simple'  ) end
function nnxtest.SpatialReSamplingEx2() template_SpatialReSamplingEx(false, 'simple'  ) end
function nnxtest.SpatialReSamplingEx3() template_SpatialReSamplingEx(false, 'average' ) end
function nnxtest.SpatialReSamplingEx4() template_SpatialReSamplingEx(true , 'bilinear') end
function nnxtest.SpatialReSamplingEx5() template_SpatialReSamplingEx(false, 'bilinear') end

function nnxtest.SpatialUpSampling()
   local fanin = math.random(1,4)
   local sizex = math.random(1,4)
   local sizey = math.random(1,4)
   local mx = math.random(2,6)
   local my = math.random(2,6)
   local module = nn.SpatialUpSampling(mx,my)
   local input = torch.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.SpatialDownSampling()
   local fanin = math.random(1,4)
   local sizex = math.random(1,4)
   local sizey = math.random(1,4)
   local mx = math.random(2,6)
   local my = math.random(2,6)
   local module = nn.SpatialDownSampling(mx,my)
   local input = torch.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.SpatialReSampling_1()
   local fanin = math.random(1,4)
   local sizex = math.random(4,8)
   local sizey = math.random(4,8)
   local osizex = math.random(2,12)
   local osizey = math.random(2,12)
   local module = nn.SpatialReSampling{owidth=osizex,oheight=osizey}
   local input = torch.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

   -- test batches (4D input)
   local batchSize = math.random(4,8)
   local input2 = torch.rand(batchSize,fanin,sizey,sizex)
   input2[2]:copy(input)
   
   local output = module:forward(input):clone()
   local output2 = module:forward(input2)
   mytester:assertTensorEq(output, output2[2], 0.00001, 'SpatialResampling batch forward err')
   
   local gradInput = module:backward(input, output):clone()
   local gradInput2 = module:backward(input2, output2)
   mytester:assertTensorEq(gradInput, gradInput2[2], 0.00001, 'SpatialResampling batch backward err')
   
   -- test rwidth/rheight
   local input = torch.randn(3,8,10)
   local module = nn.SpatialReSampling{rwidth=0.5,rheight=0.5}
   local output = module:forward(input)
   mytester:assertTableEq(output:size():totable(), {3, 4, 5}, 0.00000001, 'SpatialResampling batch rwidth/rheight err')
   
   local input = torch.randn(2,3,8,10)
   local module = nn.SpatialReSampling{rwidth=0.5,rheight=0.5}
   local output = module:forward(input)
   mytester:assertTableEq(output:size():totable(), {2, 3, 4, 5}, 0.00000001, 'SpatialResampling batch rwidth/rheight err')
end

function nnxtest.SpatialReSampling_2()
   local fanin = math.random(1,4)
   local mx = math.random()*4 + 0.1
   local my = math.random()*4 + 0.1
   local osizex = math.random(4,6)
   local osizey = math.random(4,6)
   local sizex = osizex/mx
   local sizey = osizey/my
   local module = nn.SpatialReSampling{rwidth=mx,rheight=my}
   local input = torch.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.HardShrink()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.HardShrink()

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.Abs()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Abs()

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.HardShrink()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.HardShrink()

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.SpatialConvolution()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,1)
   local sj = math.random(1,1)
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local err = nn.Jacobian.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err, precision, 'error on weight ')

   local err = nn.Jacobian.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err, precision, 'error on bias ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.Module_listModules()
   local batchSize = 4
   local inputSize, outputSize = 7, 6
   local linear = nn.Linear(inputSize, outputSize)
   local tanh = nn.Tanh()
   local reshape = nn.Reshape(outputSize/2, 2)
   local mlp3 = nn.Sequential()
   mlp3:add(linear)
   mlp3:add(tanh)
   mlp3:add(reshape)
   
   local mlp2 = nn.Sequential()
   local view = nn.View(outputSize)
   local linear2 = nn.Linear(outputSize, inputSize)
   local tanh2 = nn.Tanh()
   mlp2:add(mlp3)
   mlp2:add(view)
   mlp2:add(linear2)
   mlp2:add(tanh2)
   
   local concat = nn.ConcatTable()
   local id = nn.Identity()
   concat:add(mlp2)
   concat:add(id)
   local mlp = nn.Sequential()
   local add = nn.CAddTable()
   mlp:add(concat)
   mlp:add(add)
   
   local modules2 = {mlp, concat, mlp2, mlp3, linear, tanh, reshape, view, linear2, tanh2, id, add}
   local modules = mlp:listModules()
   
   mytester:assert(#modules2 == #modules, 'missing modules error')
   
   for i,module in ipairs(modules) do
      mytester:assert(torch.type(module) == torch.type(modules2[i]), 'module error')
   end
   
end

function nnxtest.Recurrent()
   local batchSize = 4
   local inputSize = 10
   local hiddenSize = 12
   local outputSize = 7
   local nSteps = 5 
   local inputModule = nn.Linear(inputSize, outputSize)
   local transferModule = nn.Sigmoid()
   -- test MLP feedback Module (because of Module:representations())
   local feedbackModule = nn.Sequential()
   feedbackModule:add(nn.Linear(outputSize, hiddenSize))
   feedbackModule:add(nn.Sigmoid())
   feedbackModule:add(nn.Linear(hiddenSize, outputSize))
   -- rho = nSteps
   local mlp = nn.Recurrent(outputSize, inputModule, feedbackModule, transferModule:clone(), nSteps)
   
   -- test that the internal mlps are recursable :
   local isRecursable = nn.AbstractRecurrent.isRecursable
   mytester:assert(isRecursable(mlp.initialModule, torch.randn(inputSize)), "Recurrent isRecursable() initial error")
   mytester:assert(isRecursable(mlp.recurrentModule, {torch.randn(inputSize), torch.randn(outputSize)}), "Recurrent isRecursable() recurrent error")
   
   -- test that the above test actually works
   local euclidean = nn.Euclidean(inputSize, outputSize)
   mytester:assert(not isRecursable(euclidean, torch.randn(batchSize, inputSize)), "AbstractRecurrent.isRecursable error")
   
   local gradOutputs, outputs = {}, {}
   -- inputs = {inputN, {inputN-1, {inputN-2, ...}}}}}
   local inputs
   local startModule = mlp.startModule:clone()
   inputModule = mlp.inputModule:clone()
   feedbackModule = mlp.feedbackModule:clone()
   
   local mlp6 = mlp:clone()
   mlp6:evaluate()
   
   mlp:zeroGradParameters()
   local mlp7 = mlp:clone()
   mlp7.rho = nSteps - 1
   local inputSequence = {}
   for step=1,nSteps do
      local input = torch.randn(batchSize, inputSize)
      inputSequence[step] = input
      local gradOutput
      if step ~= nSteps then
         -- for the sake of keeping this unit test simple,
         gradOutput = torch.zeros(batchSize, outputSize)
      else
         -- only the last step will get a gradient from the output
         gradOutput = torch.randn(batchSize, outputSize)
      end
      
      local output = mlp:forward(input)
      mlp:backward(input, gradOutput)
      
      local output6 = mlp6:forward(input)
      mytester:assertTensorEq(output, output6, 0.000001, "evaluation error "..step)
      
      local output7 = mlp7:forward(input)
      mlp7:backward(input, gradOutput)
      mytester:assertTensorEq(output, output7, 0.000001, "rho = nSteps-1 forward error "..step)

      table.insert(gradOutputs, gradOutput)
      table.insert(outputs, output:clone())
      
      if inputs then
         inputs = {input, inputs}
      else
         inputs = input
      end
   end
   local mlp4 = mlp:clone()
   local mlp5 = mlp:clone()
   
   -- backward propagate through time (BPTT)
   local gradInput = mlp:backwardThroughTime():clone()
   mlp:forget() -- test ability to forget
   mlp:zeroGradParameters()
   local foutputs = {}
   for step=1,nSteps do
      foutputs[step] = mlp:forward(inputSequence[step])
      mytester:assertTensorEq(foutputs[step], outputs[step], 0.00001, "Recurrent forget output error "..step)
      mlp:backward(input, gradOutputs[step])
   end
   local fgradInput = mlp:backwardThroughTime():clone()
   mytester:assertTensorEq(gradInput, fgradInput, 0.00001, "Recurrent forget gradInput error")
   
   mlp4.fastBackward = false
   local gradInput4 = mlp4:backwardThroughTime()
   mytester:assertTensorEq(gradInput, gradInput4, 0.000001, 'error slow vs fast backwardThroughTime')
   local mlp10 = mlp7:clone()
   mytester:assert(mlp10.inputs[1] == nil, 'recycle inputs error')
   mlp10:forget()
   mytester:assert(#mlp10.inputs == 4, 'forget inputs error')
   mytester:assert(#mlp10.outputs == 5, 'forget outputs error')
   local i = 0
   for k,v in pairs(mlp10.recurrentOutputs) do
      i = i + 1
   end
   mytester:assert(i == 4, 'forget recurrentOutputs error')
   
   -- rho = nSteps - 1 : shouldn't update startModule
   mlp7:backwardThroughTime()
   
   local mlp2 -- this one will simulate rho = nSteps
   local outputModules = {}
   for step=1,nSteps do
      local inputModule_ = inputModule:clone()
      local outputModule = transferModule:clone()
      table.insert(outputModules, outputModule)
      inputModule_:share(inputModule, 'weight', 'gradWeight', 'bias', 'gradBias')
      if step == 1 then
         local initialModule = nn.Sequential()
         initialModule:add(inputModule_)
         initialModule:add(startModule)
         initialModule:add(outputModule)
         mlp2 = initialModule
      else
         local parallelModule = nn.ParallelTable()
         parallelModule:add(inputModule_)
         local pastModule = nn.Sequential()
         pastModule:add(mlp2)
         local feedbackModule_ = feedbackModule:clone()
         feedbackModule_:share(feedbackModule, 'weight', 'gradWeight', 'bias', 'gradBias')
         pastModule:add(feedbackModule_)
         parallelModule:add(pastModule)
         local recurrentModule = nn.Sequential()
         recurrentModule:add(parallelModule)
         recurrentModule:add(nn.CAddTable())
         recurrentModule:add(outputModule)
         mlp2 = recurrentModule
      end
   end
   
   
   local output2 = mlp2:forward(inputs)
   mlp2:zeroGradParameters()
   
   -- unlike mlp2, mlp8 will simulate rho = nSteps -1
   local mlp8 = mlp2:clone() 
   local inputModule8 = mlp8.modules[1].modules[1]
   local m = mlp8.modules[1].modules[2].modules[1].modules[1].modules[2]
   m = m.modules[1].modules[1].modules[2].modules[1].modules[1].modules[2]
   local feedbackModule8 = m.modules[2]
   local startModule8 = m.modules[1].modules[2] -- before clone
   -- unshare the intialModule:
   m.modules[1] = m.modules[1]:clone()
   m.modules[2] = m.modules[2]:clone()
   mlp8:backward(inputs, gradOutputs[#gradOutputs])
   
   local gradInput2 = mlp2:backward(inputs, gradOutputs[#gradOutputs])
   for step=1,nSteps-1 do
      gradInput2 = gradInput2[2]
   end   
   
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "recurrent gradInput")
   mytester:assertTensorEq(outputs[#outputs], output2, 0.000001, "recurrent output")
   for step=1,nSteps do
      local output, outputModule = outputs[step], outputModules[step]
      mytester:assertTensorEq(output, outputModule.output, 0.000001, "recurrent output step="..step)
   end
   
   local mlp3 = nn.Sequential()
   -- contains params and grads of mlp2 (the MLP version of the Recurrent)
   mlp3:add(startModule):add(inputModule):add(feedbackModule)
   local params2, gradParams2 = mlp3:parameters()
   local params, gradParams = mlp:parameters()
   mytester:assert(#params2 == #params, 'missing parameters')
   mytester:assert(#gradParams == #params, 'missing gradParameters')
   for i=1,#params do
      if i > 1 then
         gradParams2[i]:div(nSteps)
      end
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.000001, 'gradParameter error ' .. i)
   end
   
   local mlp9 = nn.Sequential()
   -- contains params and grads of mlp8
   mlp9:add(startModule8):add(inputModule8):add(feedbackModule8)
   local params9, gradParams9 = mlp9:parameters()
   local params7, gradParams7 = mlp7:parameters()
   mytester:assert(#params9 == #params7, 'missing parameters')
   mytester:assert(#gradParams7 == #params7, 'missing gradParameters')
   for i=1,#params do
      if i > 1 then
         gradParams9[i]:div(nSteps-1)
      end
      mytester:assertTensorEq(gradParams7[i], gradParams9[i], 0.00001, 'gradParameter error ' .. i)
   end
   
   -- already called backwardThroughTime()
   mlp:updateParameters(0.1) 
   mlp4:updateParameters(0.1) 
   
   local params4 = mlp4:parameters()
   local params5 = mlp5:parameters()
   local params = mlp:parameters()
   mytester:assert(#params4 == #params, 'missing parameters')
   mytester:assert(#params5 == #params, 'missing parameters')
   for i=1,#params do
      mytester:assertTensorEq(params[i], params4[i], 0.000001, 'backwardThroughTime error ' .. i)
      mytester:assertTensorNe(params[i], params5[i], 0.0000000001, 'backwardThroughTime error ' .. i)
   end
   
   -- should call backwardUpdateThroughTime()
   mlp5:updateParameters(0.1)
   
   local params5 = mlp5:parameters()
   local params = mlp:parameters()
   mytester:assert(#params5 == #params, 'missing parameters')
   for i=1,#params do
      mytester:assertTensorEq(params[i], params5[i], 0.000001, 'backwardUpdateThroughTime error ' .. i)
   end
end

function nnxtest.Recurrent_TestTable()
   -- Set up RNN where internal state is a table.
   -- Trivial example is same RNN from nnxtest.Recurrent test
   -- but all layers are duplicated
   local batchSize = 4
   local inputSize = 10
   local hiddenSize = 12
   local outputSize = 7
   local nSteps = 5 
   local inputModule = nn.Linear(inputSize, outputSize)
   local transferModule = nn.Sigmoid()
   local learningRate = 0.1
   -- test MLP feedback Module
   local feedbackModule = nn.Sequential()
   feedbackModule:add(nn.Linear(outputSize, hiddenSize))
   feedbackModule:add(nn.Sigmoid())
   feedbackModule:add(nn.Linear(hiddenSize, outputSize))
   -- rho = nSteps
   local mlp = nn.Recurrent(
      nn.ParallelTable()
         :add(nn.Add(outputSize))
         :add(nn.Add(outputSize)),
      nn.ParallelTable()
         :add(inputModule:clone())
         :add(inputModule:clone()),
      nn.ParallelTable()
         :add(feedbackModule:clone())
         :add(feedbackModule:clone()),
      nn.ParallelTable()
         :add(transferModule:clone())
         :add(transferModule:clone()),
      nSteps,
      nn.ParallelTable()
         :add(nn.CAddTable())
         :add(nn.CAddTable())
   )

   local input = torch.randn(batchSize, inputSize)
   local err = torch.randn(batchSize, outputSize)
   for i=1,10 do
      mlp:forward{input, input:clone()}
      mlp:backward({input, input:clone()}, {err, err:clone()})
   end
   mlp:backwardThroughTime(learningRate)
end

function nnxtest.LSTM()
   local batchSize = math.random(1,2)
   local inputSize = math.random(3,4)
   local outputSize = math.random(5,6)
   local nStep = 3
   local input = {}
   local gradOutput = {}
   for step=1,nStep do
      input[step] = torch.randn(batchSize, inputSize)
      if step == nStep then
         -- for the sake of keeping this unit test simple,
         gradOutput[step] = torch.randn(batchSize, outputSize)
      else
         -- only the last step will get a gradient from the output
         gradOutput[step] = torch.zeros(batchSize, outputSize)
      end
   end
   local lstm = nn.LSTM(inputSize, outputSize)
   
   local isRecursable = nn.AbstractRecurrent.isRecursable
   local inputTable = {torch.randn(batchSize, inputSize), torch.randn(batchSize, outputSize), torch.randn(batchSize, outputSize)}
   mytester:assert(isRecursable(lstm.recurrentModule, inputTable), "LSTM isRecursable() error")
   
   -- we will use this to build an LSTM step by step (with shared params)
   local lstmStep = lstm.recurrentModule:clone()
   
   -- forward/backward through LSTM
   local output = {}
   lstm:zeroGradParameters()
   for step=1,nStep do
      output[step] = lstm:forward(input[step])
      assert(torch.isTensor(input[step]))
      lstm:backward(input[step], gradOutput[step], 1)
   end   
   local gradInput = lstm:backwardThroughTime()
   
   local mlp2 -- this one will simulate rho = nSteps
   local inputs
   for step=1,nStep do
      -- iteratively build an LSTM out of non-recurrent components
      local lstm = lstmStep:clone()
      lstm:share(lstmStep, 'weight', 'gradWeight', 'bias', 'gradBias')
      if step == 1 then
         mlp2 = lstm
      else
         local rnn = nn.Sequential()
         local para = nn.ParallelTable()
         para:add(nn.Identity()):add(mlp2)
         rnn:add(para)
         rnn:add(nn.FlattenTable())
         rnn:add(lstm)
         mlp2 = rnn
      end
      
      -- prepare inputs for mlp2
      if inputs then
         inputs = {input[step], inputs}
      else
         inputs = {input[step], torch.zeros(batchSize, outputSize), torch.zeros(batchSize, outputSize)}
      end
   end
   mlp2:add(nn.SelectTable(1)) --just output the output (not cell)
   local output2 = mlp2:forward(inputs)
   
   mlp2:zeroGradParameters()
   local gradInput2 = mlp2:backward(inputs, gradOutput[nStep], 1/nStep)
   mytester:assertTensorEq(gradInput2[2][2][1], gradInput, 0.00001, "LSTM gradInput error")
   mytester:assertTensorEq(output[nStep], output2, 0.00001, "LSTM output error")
   
   local params, gradParams = lstm:parameters()
   local params2, gradParams2 = lstmStep:parameters()
   mytester:assert(#params == #params2, "LSTM parameters error "..#params.." ~= "..#params2)
   for i, gradParam in ipairs(gradParams) do
      local gradParam2 = gradParams2[i]
      mytester:assertTensorEq(gradParam, gradParam2, 0.000001, 
         "LSTM gradParam "..i.." error "..tostring(gradParam).." "..tostring(gradParam2))
   end
   
   gradParams = lstm.recursiveCopy(nil, gradParams)
   gradInput = gradInput:clone()
   mytester:assert(lstm.zeroTensor:sum() == 0, "zeroTensor error")
   lstm:forget()
   output = lstm.recursiveCopy(nil, output)
   local output3 = {}
   lstm:zeroGradParameters()
   for step=1,nStep do
      output3[step] = lstm:forward(input[step])
      lstm:backward(input[step], gradOutput[step], 1)
   end   
   local gradInput3 = lstm:updateGradInputThroughTime()
   lstm:accGradParametersThroughTime()
   
   mytester:assert(#output == #output3, "LSTM output size error")
   for i,output in ipairs(output) do
      mytester:assertTensorEq(output, output3[i], 0.00001, "LSTM forget (updateOutput) error "..i)
   end
   
   mytester:assertTensorEq(gradInput, gradInput3, 0.00001, "LSTM updateGradInputThroughTime error")
   --if true then return end
   local params3, gradParams3 = lstm:parameters()
   mytester:assert(#params == #params3, "LSTM parameters error "..#params.." ~= "..#params3)
   for i, gradParam in ipairs(gradParams) do
      local gradParam3 = gradParams3[i]
      mytester:assertTensorEq(gradParam, gradParam3, 0.000001, 
         "LSTM gradParam "..i.." error "..tostring(gradParam).." "..tostring(gradParam3))
   end
end

function nnxtest.Sequencer()
   local batchSize = 4
   local inputSize = 10
   local outputSize = 7
   local nSteps = 5 
   local inputModule = nn.Linear(inputSize, outputSize)
   local transferModule = nn.Sigmoid()
   -- test MLP feedback Module (because of Module:representations())
   local feedbackModule = nn.Linear(outputSize, outputSize)
   -- rho = nSteps
   local rnn = nn.Recurrent(outputSize, inputModule, feedbackModule, transferModule, nSteps)
   local rnn2 = rnn:clone()
   
   local inputs, outputs, gradOutputs = {}, {}, {}
   for step=1,nSteps do
      inputs[step] = torch.randn(batchSize, inputSize)
      outputs[step] = rnn:forward(inputs[step])
      gradOutputs[step] = torch.randn(batchSize, outputSize)
      rnn:backward(inputs[step], gradOutputs[step])
   end
   rnn:backwardThroughTime()
   
   local rnn3 = nn.Sequencer(rnn2)
   local outputs3 = rnn3:forward(inputs)
   local gradInputs3 = rnn3:backward(inputs, gradOutputs)
   mytester:assert(#outputs3 == #outputs, "Sequencer output size err")
   mytester:assert(#gradInputs3 == #rnn.gradInputs, "Sequencer gradInputs size err")
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs3[step], output, 0.00001, "Sequencer output "..step)
      mytester:assertTensorEq(gradInputs3[step], rnn.gradInputs[step], 0.00001, "Sequencer gradInputs "..step)
   end
end

function nnxtest.Repeater()
   local batchSize = 4
   local inputSize = 10
   local outputSize = 7
   local nSteps = 5 
   local inputModule = nn.Linear(inputSize, outputSize)
   local transferModule = nn.Sigmoid()
   -- test MLP feedback Module (because of Module:representations())
   local feedbackModule = nn.Linear(outputSize, outputSize)
   -- rho = nSteps
   local rnn = nn.Recurrent(outputSize, inputModule, feedbackModule, transferModule, nSteps)
   local rnn2 = rnn:clone()
   
   local inputs, outputs, gradOutputs = {}, {}, {}
   local input = torch.randn(batchSize, inputSize)
   for step=1,nSteps do
      outputs[step] = rnn:forward(input)
      gradOutputs[step] = torch.randn(batchSize, outputSize)
      rnn:backward(input, gradOutputs[step])
   end
   rnn:backwardThroughTime()
   
   local rnn3 = nn.Repeater(rnn2, nSteps)
   local outputs3 = rnn3:forward(input)
   local gradInput3 = rnn3:backward(input, gradOutputs)
   mytester:assert(#outputs3 == #outputs, "Repeater output size err")
   mytester:assert(#outputs3 == #rnn.gradInputs, "Repeater gradInputs size err")
   local gradInput = rnn.gradInputs[1]:clone():zero()
   for step,output in ipairs(outputs) do
      mytester:assertTensorEq(outputs3[step], output, 0.00001, "Sequencer output "..step)
      gradInput:add(rnn.gradInputs[step])
   end
   mytester:assertTensorEq(gradInput3, gradInput, 0.00001, "Repeater gradInput err")
end

function nnxtest.SpatialNormalization_Gaussian2D()
   local inputSize = math.random(11,20)
   local kersize = 9
   local nbfeatures = math.random(5,10)
   local kernel = image.gaussian(kersize)
   local module = nn.SpatialNormalization(nbfeatures,kernel,0.1)
   local input = torch.rand(nbfeatures,inputSize,inputSize)
   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
end

function nnxtest.SpatialNormalization_Gaussian1D()
   local inputSize = math.random(14,20)
   local kersize = 15
   local nbfeatures = math.random(5,10)
   local kernelv = image.gaussian1D(11):resize(11,1)
   local kernelh = kernelv:t()
   local module = nn.SpatialNormalization(nbfeatures, {kernelv,kernelh}, 0.1)
   local input = torch.rand(nbfeatures,inputSize,inputSize)
   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
end

function nnxtest.SpatialNormalization_io()
   local inputSize = math.random(11,20)
   local kersize = 7
   local nbfeatures = math.random(2,5)
   local kernel = image.gaussian(kersize)
   local module = nn.SpatialNormalization(nbfeatures,kernel)
   local input = torch.rand(nbfeatures,inputSize,inputSize)
   local ferr, berr = nn.Jacobian.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

local function template_SpatialFovea(fx,fy,bilinear)
   local channels = math.random(1,4)
   local iwidth = 16
   local iheight = 16

   local module = nn.SpatialFovea{nInputPlane = channels,
                                  ratios = {1,2},
                                  preProcessors = {nn.Identity(),
                                                   nn.Identity()},
                                  processors = {nn.SpatialConvolution(channels,4,3,3),
                                                nn.SpatialConvolution(channels,4,3,3)},
                                  bilinear = bilinear,
                                  fov = 3,
                                  sub = 1}

   input = torch.rand(channels, iheight, iwidth)

   module:focus(fx,fy,3)
   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   module:focus()
   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end
function nnxtest.SpatialFovea_focused() template_SpatialFovea(4,7) end
function nnxtest.SpatialFovea_unfocused() template_SpatialFovea() end
function nnxtest.SpatialFovea_bilinear() template_SpatialFovea(nil,nil,true) end

local function template_SpatialPyramid(fx,fy)
   local channels = math.random(1,4)
   local iwidth = 16
   local iheight = 16

   local pyr = nn.SpatialPyramid({1,2},{nn.SpatialConvolution(channels,4,3,3),
				       nn.SpatialConvolution(channels,4,3,3)},
				 3, 3, 1, 1)
   local module = nn.Sequential()
   module:add(pyr)
   module:add(nn.JoinTable(1))

   input = torch.rand(channels, iheight, iwidth)

   pyr:focus(fx,fy,3,3)
   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   pyr:focus()
   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.SpatialPyramid_focused() template_SpatialPyramid(5,3) end
function nnxtest.SpatialPyramid_unfocused() template_SpatialPyramid() end

local function template_SpatialGraph(channels, iwidth, iheight, dist, norm)
   local module = nn.SpatialGraph{normalize=norm, dist=dist}
   local input = torch.rand(iwidth, iheight, channels)
   local err = nn.Jacobian.testJacobian(module, input, 0.1, 1)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end
function nnxtest.SpatialGraph_1() template_SpatialGraph(3, 16, 16, 'euclid', true) end
function nnxtest.SpatialGraph_2() template_SpatialGraph(16, 4, 4, 'euclid', true) end
function nnxtest.SpatialGraph_3() template_SpatialGraph(256, 2, 2, 'euclid', false) end
function nnxtest.SpatialGraph_4() template_SpatialGraph(2, 16, 16, 'cosine', false) end
function nnxtest.SpatialGraph_5() template_SpatialGraph(64, 3, 3, 'cosine', false) end

local function template_SpatialMatching(channels, iwidth, iheight, maxw, maxh, full_output)
   local module = nn.Sequential()
   module:add(nn.SplitTable(1))
   local parallel = nn.ParallelTable()
   local seq1 = nn.Sequential()
   seq1:add(nn.Narrow(2, math.floor(maxh/2), iheight-maxh+1))
   seq1:add(nn.Narrow(3, math.floor(maxw/2), iwidth -maxw+1))
   parallel:add(seq1)
   parallel:add(nn.Identity())
   module:add(parallel)
   module:add(nn.SpatialMatching(maxh, maxw, full_output))
   local input = torch.rand(2, channels, iheight, iwidth)
   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
   
   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end
function nnxtest.SpatialMatching_1() template_SpatialMatching(4, 16, 16, 5, 5, true) end
function nnxtest.SpatialMatching_2() template_SpatialMatching(4, 16, 16, 5, 5, false) end
function nnxtest.SpatialMatching_3() template_SpatialMatching(3, 16, 16, 6, 6, true) end
function nnxtest.SpatialMatching_4() template_SpatialMatching(3, 20, 20, 4, 4, false) end
function nnxtest.SpatialMatching_5() template_SpatialMatching(3, 12, 16, 5, 7, true) end
--function nnxtest.SpatialMatching_6() template_SpatialMatching(4, 16, 32, 9, 5, false) end

function nnxtest.SoftMaxTree()
   local input = torch.randn(5,100)
   local target = torch.IntTensor{20,23,27,10,8}
   local grad = torch.randn(5)
   local root_id = 29
   local hierarchy={
      [29]=torch.IntTensor{30,1,2}, [1]=torch.IntTensor{3,4,5}, 
      [2]=torch.IntTensor{6,7,8}, [3]=torch.IntTensor{9,10,11},
      [4]=torch.IntTensor{12,13,14}, [5]=torch.IntTensor{15,16,17},
      [6]=torch.IntTensor{18,19,20}, [7]=torch.IntTensor{21,22,23},
      [8]=torch.IntTensor{24,25,26,27,28}
   }
   local smt = nn.SoftMaxTree(100, hierarchy, root_id)
   smt:zeroGradParameters()
   -- compare to the inefficient version for example 3
   local concat = nn.ConcatTable()
   local indices = {3,3,4}
   local parentIds = {29,2,8}
   local linears = {}
   
   for i,parentId in ipairs(parentIds) do
      local s = nn.Sequential()
      local linear = nn.Linear(100,hierarchy[parentId]:size(1))
      linears[parentId] = linear
      local param, grad = smt:getNodeParameters(parentId)
      local weight, bias = unpack(param)
      local gradWeight, gradBias = unpack(grad)
      mytester:asserteq(gradWeight:sum(), 0, 0.000001)
      mytester:asserteq(gradBias:sum(), 0, 0.000001)
      linear.weight:set(weight:clone())
      linear.bias:set(bias:clone())
      s:add(linear)
      s:add(nn.LogSoftMax())
      s:add(nn.Narrow(1,indices[i],1))
      concat:add(s)
   end
   local mlp = nn.Sequential()
   mlp:add(concat)
   mlp:add(nn.CAddTable())
   -- will fail without this:
   smt:zeroGradParameters()
   mlp:zeroGradParameters()
   -- forward backward
   local output = smt:forward{input, target}
   local mlp_act = mlp:forward(input[3])
   local gradInput = smt:backward({input, target}, grad)[1]
   local mlp_grad = mlp:backward(input[3], grad:narrow(1,3,1))
   -- compare
   mytester:assert(math.abs(output[3] - mlp_act[1]) < 0.00001)
   mytester:assertTensorEq(gradInput[3], mlp_grad, 0.00001)
   -- update
   mytester:asserteq(smt.updates[29], 5, 0.000001)
   smt:updateParameters(0.1)
   mlp:updateParameters(0.1)
   local parentId = 8
   local param, grads = smt:getNodeParameters(parentId)
   local weight, bias = unpack(param)
   local gradWeight, gradBias = unpack(grads)
   local linear = linears[parentId]
   mytester:assertTensorEq(weight, linear.weight, 0.000001)
   mytester:assertTensorEq(gradWeight, linear.gradWeight, 0.000001)
   mytester:assertTensorEq(bias, linear.bias, 0.000001)
   mytester:assertTensorEq(gradBias, linear.gradBias, 0.000001)
   -- sharedClone
   local smt2 = smt:sharedClone()
   output = smt:forward{input, target}
   local output2 = smt2:forward{input, target}
   mytester:assertTensorEq(output, output2, 0.00001)
   -- accUpdate
   local smt3 = nn.SoftMaxTree(100, hierarchy, root_id, true)
   smt3:zeroGradParameters()
   smt3.weight = smt.weight:clone()
   smt3.bias = smt.bias:clone()
   local output3 = smt3:forward{input, target}
   local output = smt3:forward{input, target}
   local gradInput3 = smt3:backwardUpdate({input, target}, grad, 0.1)[1]
   local gradInput = smt:backwardUpdate({input, target}, grad, 0.1)[1]
   mytester:assertTensorEq(output3, output, 0.00001)
   mytester:assertTensorEq(gradInput3, gradInput, 0.00001)
   local parentId = 8
   local weight3, bias3 = unpack(smt3:getNodeParameters(parentId))
   local params = smt:getNodeParameters(parentId)
   local weight, bias = unpack(params)
   mytester:assertTensorEq(weight3, weight, 0.000001)
   mytester:assertTensorEq(bias3, bias, 0.000001)
end

function nnxtest.TreeNLLCriterion()
   local input = torch.randn(5,10)
   local target = torch.ones(5) --all targets are 1
   local c = nn.TreeNLLCriterion() 
   -- the targets are actually ignored (SoftMaxTree uses them before TreeNLLCriterion)
   local err = c:forward(input, target)
   gradInput = c:backward(input, target)
   -- compare to ClassNLLCriterion
   local c2 = nn.ClassNLLCriterion()
   local err2 = c2:forward(input, target)
   local gradInput2 = c2:backward(input, target)
   mytester:asserteq(err2, err, 0.00001)
   mytester:assertTensorEq(gradInput2:narrow(2,1,1), gradInput, 0.00001)
end

local function blur(mean, stdv, size)
   local range = torch.range(1,size):float()
   local a = 1/(stdv*math.sqrt(2*math.pi))
   local b = -1/(2*stdv*stdv)
   return range:add(-mean):pow(2):mul(b):exp():mul(a)
end

function nnxtest.Balance()
   local inputSize = 7 
   local batchSize = 3
   local nBatch = 1
   
   local input = torch.randn(batchSize, inputSize):mul(0.1):float()
   for i=1,batchSize do
      input[i]:add(blur(3, 1, inputSize):float())
   end
   local sm = nn.SoftMax()
   sm:float()
   input = sm:forward(input)
   local gradOutput = torch.randn(batchSize, inputSize):float()
   local bl = nn.Balance(nBatch)
   bl:float()
   
   local output = bl:forward(input)
   local p_y = output:sum(1):div(output:sum())
   mytester:assert(p_y:std() < 0.02)
   mytester:assert(math.abs(p_y:sum() - 1) < 0.000001)
   
   local gradInput = bl:backward(input, gradOutput)
end

function nnxtest.NarrowLookupTable()
   local nIndex = 5
   local dictSize = 100
   local batchSize = 8
   local embedSize = 32
   local deltaSize = 4
   local lr = 0.1
   
   -- 1D input ascDelta = false
   local input = torch.randperm(dictSize):narrow(1,1,nIndex)
   local nlt = nn.NarrowLookupTable(deltaSize, dictSize, embedSize, false)
   local output = nlt:forward(input)
   
   local output2 = torch.Tensor(120):zero()
   local narrowSize = embedSize
   local idx = 121 - narrowSize
   for i=nIndex,1,-1 do
      output2:narrow(1, idx, narrowSize):copy(nlt.weight[input[i]]:narrow(1,1,narrowSize))
      narrowSize = narrowSize - deltaSize
      idx = idx - narrowSize
   end
   mytester:assertTensorEq(output, output2, 0.000001, "1D forward ascDelta = false error")
   
   nlt:zeroGradParameters()
   local gradWeight2 = nlt.gradWeight:clone()
   nlt:backward(input, output)
   local narrowSize = embedSize
   local idx = 121 - narrowSize
   for i=nIndex,1,-1 do
      gradWeight2[input[i]]:narrow(1, 1, narrowSize):add(output:narrow(1,idx,narrowSize))
      narrowSize = narrowSize - deltaSize
      idx = idx - narrowSize
   end
   mytester:assertTensorEq(nlt.gradWeight, gradWeight2, 0.000001, "1D backward ascDelta = false error")
   
   -- 1D input
   local input = torch.randperm(dictSize):narrow(1,1,nIndex)
   local nlt = nn.NarrowLookupTable(deltaSize, dictSize, embedSize)
   local output = nlt:forward(input)
   
   local output2 = torch.Tensor(120):zero()
   local narrowSize = embedSize
   local idx = 1
   for i=1,nIndex do
      output2:narrow(1, idx, narrowSize):copy(nlt.weight[input[i]]:narrow(1,1,narrowSize))
      idx = idx + narrowSize
      narrowSize = narrowSize - deltaSize
   end
   mytester:assertTensorEq(output, output2, 0.000001, "1D forward error")
   
   nlt:zeroGradParameters()
   local gradWeight2 = nlt.gradWeight:clone()
   nlt:backward(input, output)
   local idx = 1
   local narrowSize = embedSize
   for i=1,nIndex do
      gradWeight2[input[i]]:narrow(1, 1, narrowSize):add(output:narrow(1,idx,narrowSize))
      idx = idx + narrowSize
      narrowSize = narrowSize - deltaSize
   end
   mytester:assertTensorEq(nlt.gradWeight, gradWeight2, 0.000001, "1D backward error")
   
   nlt:zeroGradParameters()
   local weight2 = nlt.weight:clone()
   nlt:backwardUpdate(input, output, lr)
   local idx = 1
   local narrowSize = embedSize
   for i=1,nIndex do
      weight2[input[i]]:narrow(1, 1, narrowSize):add(-lr, output:narrow(1,idx,narrowSize))
      idx = idx + narrowSize
      narrowSize = narrowSize - deltaSize
   end
   mytester:assertTensorEq(nlt.weight, weight2, 0.000001, "1D backwardUpdate error")
   
   -- 2D input
   nlt:float()
   local input = torch.randperm(dictSize):narrow(1,1,nIndex*batchSize):view(8,-1)
   local output = nlt:forward(input)
   local output2 = torch.FloatTensor(batchSize, 120):zero()
   for k=1,batchSize do
      local input = input[k]
      local output2 = output2[k]
      local narrowSize = embedSize
      local idx = 1
      for i=1,nIndex do
         output2:narrow(1, idx, narrowSize):add(nlt.weight[input[i]]:narrow(1,1,narrowSize))
         idx = idx + narrowSize
         narrowSize = narrowSize - deltaSize
      end
   end
   mytester:assertTensorEq(output, output2, 0.000001, "2D forward error")
   
   nlt:zeroGradParameters()
   local gradWeight2 = nlt.gradWeight:clone()
   nlt:backward(input, output)
   for k=1,batchSize do
      local input = input[k]
      local output = output[k]
      local idx = 1
      local narrowSize = embedSize
      for i=1,nIndex do
         gradWeight2[input[i]]:narrow(1,1,narrowSize):add(output:narrow(1,idx,narrowSize))
         idx = idx + narrowSize
         narrowSize = narrowSize - deltaSize
      end
   end
   mytester:assertTensorEq(nlt.gradWeight, gradWeight2, 0.000001, "2D backward error")
   
   nlt:zeroGradParameters()
   local weight2 = nlt.weight:clone()
   nlt:backwardUpdate(input, output, lr)
   for k=1,batchSize do
      local input = input[k]
      local output = output[k]
      local idx = 1
      local narrowSize = embedSize
      for i=1,nIndex do
         weight2[input[i]]:narrow(1,1,narrowSize):add(-lr, output:narrow(1,idx,narrowSize))
         idx = idx + narrowSize
         narrowSize = narrowSize - deltaSize
      end
   end
   mytester:assertTensorEq(nlt.weight, weight2, 0.000001, "2D backwardUpdate error")
end

function nnxtest.MultiSoftMax()
   local inputSize = 7 
   local nSoftmax = 5
   local batchSize = 3
   
   local input = torch.randn(batchSize, nSoftmax, inputSize)
   local gradOutput = torch.randn(batchSize, nSoftmax, inputSize)
   local msm = nn.MultiSoftMax()
   
   local output = msm:forward(input)
   local gradInput = msm:backward(input, gradOutput)
   mytester:assert(output:isSameSizeAs(input))
   mytester:assert(gradOutput:isSameSizeAs(gradInput))
   
   local sm = nn.SoftMax()
   local input2 = input:view(batchSize*nSoftmax, inputSize)
   local output2 = sm:forward(input2)
   local gradInput2 = sm:backward(input2, gradOutput:view(batchSize*nSoftmax, inputSize))
   
   mytester:assertTensorEq(output, output2, 0.000001)
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001)
end

function nnxtest.PushPullTable()
   -- use for targets with SoftMaxTree
   local input = torch.randn(5,50)
   local target = torch.IntTensor{20,23,27,10,8}
   local gradOutput = torch.randn(5)
   local root_id = 29
   local hierarchy={
      [29]=torch.IntTensor{30,1,2}, [1]=torch.IntTensor{3,4,5}, 
      [2]=torch.IntTensor{6,7,8}, [3]=torch.IntTensor{9,10,11},
      [4]=torch.IntTensor{12,13,14}, [5]=torch.IntTensor{15,16,17},
      [6]=torch.IntTensor{18,19,20}, [7]=torch.IntTensor{21,22,23},
      [8]=torch.IntTensor{24,25,26,27,28}
   }
   local smt = nn.SoftMaxTree(100, hierarchy, root_id)
   -- create a network where inputs are fed through softmaxtree 
   -- and targets are teleported (pushed then pulled) to softmaxtree
   local mlp = nn.Sequential()
   local linear = nn.Linear(50,100)
   local push = nn.PushTable(2)
   local pull = push:pull(2)
   mlp:add(push)
   mlp:add(nn.SelectTable(1))
   mlp:add(linear)
   mlp:add(pull)
   mlp:add(smt)
   -- compare to simpler alternative
   local mlp2 = nn.Sequential()
   local para = nn.ParallelTable()
   para:add(linear:clone())
   para:add(nn.Identity())
   mlp2:add(para)
   mlp2:add(smt:clone())
   local inputTable = {input, target}
   local output = mlp:forward(inputTable)
   local output2 = mlp2:forward(inputTable)
   local gradInput = mlp:backward(inputTable, gradOutput)
   local gradInput2 = mlp2:backward(inputTable, gradOutput)
   mytester:assertTensorEq(output, output2, 0.00001, "push/pull forward error")
   mytester:assertTensorEq(gradInput[1], gradInput[1], 0.00001, "push/pull backward error")
   mytester:assertTensorEq(gradInput[2], gradInput[2], 0.00001, "push/pull backward error")
   
   -- test multi-pull case
   local mlp = nn.Sequential()
   local push = nn.PushTable(2)
   mlp:add(push)
   mlp:add(nn.Identity())
   mlp:add(push:pull(2))
   mlp:add(push:pull(3))
   mlp:add(push:pull(1))
   -- {1,2} -> {2,1,2,2}
   local output = mlp:forward(inputTable)
   mytester:assertTensorEq(output[1], inputTable[2], 0.00001, "push/pull multi-forward error")
   mytester:assertTensorEq(output[2], inputTable[1], 0.00001, "push/pull multi-forward error")
   mytester:assertTensorEq(output[3], inputTable[2], 0.00001, "push/pull multi-forward error")
   mytester:assertTensorEq(output[4], inputTable[2], 0.00001, "push/pull multi-forward error")
   local gradOutput = {inputTable[2]:clone(), inputTable[1]:clone(), inputTable[2]:clone(), inputTable[2]:clone()}
   local gradInput = mlp:backward(inputTable, gradOutput)
   local gradInput2 = inputTable[2]:clone():mul(3) 
   mytester:assertTensorEq(gradInput[1], gradInput[1], 0.00001, "push/pull multi-backward error")
   mytester:assertTensorEq(gradInput[2], gradInput[2], 0.00001, "push/pull multi-backward error")
end

function nnx.test(tests)
   xlua.require('image',true)
   mytester = torch.Tester()
   mytester:add(nnxtest)
   math.randomseed(os.time())
   mytester:run(tests)
end
