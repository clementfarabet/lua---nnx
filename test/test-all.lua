
local nnxtest = {}
local precision = 1e-5
local mytester

function nnxtest.SpatialPadding()
   local fanin = math.random(1,3)
   local sizex = math.random(4,16)
   local sizey = math.random(4,16)
   local pad_l = math.random(0,8)
   local pad_r = math.random(0,8)
   local pad_t = math.random(0,8)
   local pad_b = math.random(0,8)
   local module = nn.SpatialPadding(pad_l, pad_r, pad_t, pad_b)
   local input = lab.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.SpatialLinear()
   local fanin = math.random(1,10)
   local fanout = math.random(1,10)
   local in1 = lab.rand(fanin,1,1)
   local module = nn.SpatialLinear(fanin,fanout)
   local moduleg = nn.Linear(fanin,fanout)
   moduleg.weight:copy(module.weight)
   moduleg.bias:copy(module.bias)
   local out = module:forward(in1)
   local ground = moduleg:forward(in1:select(2,1,1):select(2,1,1))
   local err = out:dist(ground)
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local fanin = math.random(1,10)
   local fanout = math.random(1,10)
   local sizex = math.random(4,16)
   local sizey = math.random(4,16)
   local module = nn.SpatialLinear(fanin, fanout)
   local input = lab.rand(fanin,sizey,sizex)

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
   local input = lab.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.SpatialUpSampling()
   local fanin = math.random(1,4)
   local sizex = math.random(1,4)
   local sizey = math.random(1,4)
   local mx = math.random(2,6)
   local my = math.random(2,6)
   local module = nn.SpatialUpSampling(mx,my)
   local input = lab.rand(fanin,sizey,sizex)

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
   local module = nn.SpatialReSampling(nil,nil,osizex,osizey)
   local input = lab.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.SpatialReSampling_2()
   local fanin = math.random(1,4)
   local mx = math.random()*4 + 0.1
   local my = math.random()*4 + 0.1
   local osizex = math.random(4,6)
   local osizey = math.random(4,6)
   local sizex = osizex/mx
   local sizey = osizey/my
   local module = nn.SpatialReSampling(mx,my)
   local input = lab.rand(fanin,sizey,sizex)

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.Power()
   local in1 = lab.rand(10,20)
   local module = nn.Power(2)
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local pw = random.uniform()*math.random(1,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Power(pw)

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.Square()
   local in1 = lab.rand(10,20)
   local module = nn.Square()
   local out = module:forward(in1)
   local err = out:dist(in1:cmul(in1))
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Square()

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.Sqrt()
   local in1 = lab.rand(10,20)
   local module = nn.Sqrt()
   local out = module:forward(in1)
   local err = out:dist(in1:sqrt())
   mytester:asserteq(err, 0, torch.typename(module) .. ' - forward err ')

   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Sqrt()

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input, 0.1, 2)
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

function nnxtest.SpatialLogSoftMax()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.SpatialLogSoftMax()

   local err = nn.Jacobian.testJacobian(module, input, 0.1, 2)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input, 0.1, 2)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
end

function nnxtest.Threshold()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local ink = math.random(5,10)
   local input = torch.Tensor(ink, inj, ini):zero()

   local module = nn.Threshold(random.uniform(-2,2),random.uniform(-2,2))

   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = nn.Jacobian.testIO(module, input)
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

function nnxtest.SpatialConvolutionTable_1()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,1)
   local sj = math.random(1,1)

   local ct = nn.tables.full(from,to)
   local module = nn.SpatialConvolutionTable(ct, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()
   module:reset()

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

function nnxtest.SpatialConvolutionTable_2()
   local from = math.random(1,10)
   local to = math.random(1,10)
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,1)
   local sj = math.random(1,1)

   local ct = nn.tables.oneToOne(from)
   local module = nn.SpatialConvolutionTable(ct, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()
   module:reset()

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

function nnxtest.SpatialConvolutionTable_3()
   local from = math.random(2,6)
   local to = math.random(4,8)
   local ini = math.random(10,20)
   local inj = math.random(10,20)
   local ki = math.random(1,10)
   local kj = math.random(1,10)
   local si = math.random(1,1)
   local sj = math.random(1,1)

   local ct = nn.tables.random(from,to,from-1)
   local module = nn.SpatialConvolutionTable(ct, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()
   module:reset()

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

function nnxtest.SpatialNormalization_Gaussian2D()
   local inputSize = math.random(11,20)
   local kersize = 9
   local nbfeatures = math.random(5,10)
   local kernel = image.gaussian(kersize)
   local module = nn.SpatialNormalization(nbfeatures,kernel,0.1)
   local input = lab.rand(nbfeatures,inputSize,inputSize)
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
   local input = lab.rand(nbfeatures,inputSize,inputSize)
   local err = nn.Jacobian.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
end

function nnxtest.SpatialNormalization_io()
   local inputSize = math.random(11,20)
   local kersize = 7
   local nbfeatures = math.random(2,5)
   local kernel = image.gaussian(kersize)
   local module = nn.SpatialNormalization(nbfeatures,kernel)
   local input = lab.rand(nbfeatures,inputSize,inputSize)
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

   input = lab.rand(channels, iheight, iwidth)

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

local function template_SpatialGraph(channels, iwidth, iheight, dist, norm)
   local module = nn.SpatialGraph{normalize=norm, dist=dist}
   local input = lab.rand(iwidth, iheight, channels)
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

function nnx.test()
   xlua.require('image',true)
   mytester = torch.Tester()
   mytester:add(nnxtest)
   math.randomseed(os.time())
   mytester:run()
end
