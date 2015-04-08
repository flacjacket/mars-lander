require 'nn'
require 'image'
require 'underscore'

convNet = {}
convNet.__index = convNet
setmetatable(convNet, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function convNet.new(opt)
  local self = setmetatable({},convNet)
  self.net=nn.Sequential()
  for key, value in pairs(opt) do
    self[key] = value
  end
  return self
end

function convNet:getNet()
  return self.net
end
  

-- in Lua functions are first class citizens
local trans_layer ={}
trans_layer['Tanh']=nn.Tanh
trans_layer['ReLU']=nn.ReLU
trans_layer['Sigmoid']=nn.Sigmoid

-- Example build
-- parameters,gradParameters = myNet:build({1,64,64,128,#species}, 2, 5)
function convNet:build()  -- HARD CODED TEST

	self.net:add(nn.SplitTable(3))
	local parallelNets = nn.ParallelTable()
	for i = 1,16 do  -- 16 = number of radial components
		local parallelSequence = nn.Sequential()
		parallelSequence:add(nn.Reshape(16+4,1))  -- 16+4 = number of buffered angular components
		parallelSequence:add(nn.SpatialConvolutionMM(1,8,1,5,1,1))  -- note kW (nonintuitively) comes before kH on parameter list
		parallelSequence:add(nn.Tanh())
		parallelSequence:add(nn.SpatialMaxPooling(1,2,1,2))  -- angular pooling
		parallelNets:add(parallelSequence)
	end
	self.net:add(parallelNets)
	self.net:add(nn.JoinTable(3)) -- This randomly puts toegether maps which have experienced different filters.  Just skip to fully connected?
	self.net:add(nn.SpatialConvolutionMM(8,16,5,8,1,1))
	self.net:add(nn.Tanh())
	self.net:add(nn.SpatialMaxPooling(2,1,2,1))  -- radial pooling


	self.net:add(nn.View(96))  -- 16*12/2
	self.net:add(nn.Linear(96,96))
	self.net:add(nn.Tanh())
	self.net:add(nn.Dropout(0.4))	
	self.net:add(nn.Linear(96,2))
	self.net:add(nn.LogSoftMax())
	
	self:reset()
end
	


--[[
function convNet:build(dimensions, kW, dW, pools)
  
  --local normkernel = image.gaussian1D(3)
  print('These dimensions should be integers. If not then you need to add padding')
  --   input to first layer
  self.net:add(nn.SpatialConvolutionMM(1, dimensions[1], kW[1].azimuth, kW[1].radial, dW[1].azimuth, dW[1].radial))
  local output_width = (self.width - kW[1].width)/dW[1].width + 1
  local output_height = (self.width - kW[1].width)/dW[1].width + 1
  self.net:add(trans_layer[self.transfer]())
  self.net:add(nn.SpatialMaxPooling(dimensions[1],2,pools[1],pools[1],pools[1],pools[1]))
  owidth = owidth/pools[1]
  --self.net:add(nn.SpatialSubtractiveNormalization(dimensions[1], normkernel))
  self.net:add(nn.Dropout(self.dropout[1])) 
  print (owidth..' layer 1 output: width and height')
  print (owidth*owidth*dimensions[1]..' number of features')

  --   first to second layer
  self.net:add(nn.SpatialConvolutionMM(dimensions[1], dimensions[2], kW[2], kW[2], dW[2], dW[2]))
  owidth = (owidth - kW[2])/dW[2] + 1
  self.net:add(trans_layer[self.transfer]())
  self.net:add(nn.SpatialLPPooling(dimensions[2],2,pools[2],pools[2],pools[2],pools[2]))
  owidth = owidth/pools[2]
  --self.net:add(nn.SpatialSubtractiveNormalization(dimensions[2], normkernel))
  --self.net:add(nn.Dropout(self.dropout[2])) 
  print (owidth..' layer 2 output: width and height')
  print (owidth*owidth*dimensions[2]..' layer 2: number of features')
  local  linearFeatDim = owidth*owidth*dimensions[2]
  
-- --   second layer to linear
  if #dimensions==3 then
    self.net:add(nn.SpatialConvolutionMM(dimensions[2], dimensions[3], kW[3], kW[3], dW[3], dW[3]))
    owidth = (owidth - kW[3])/dW[3] + 1
    self.net:add(trans_layer[self.transfer]())
    self.net:add(nn.SpatialMaxPooling(dimensions[3],2,pools[3],pools[3],pools[3],pools[3]))
    owidth = owidth/pools[3]
    --self.net:add(nn.SpatialSubtractiveNormalization(dimensions[3], normkernel))
    --self.net:add(nn.Dropout(self.dropout[3])) 
    print (owidth..' layer 3 output: width and height')
    print (owidth*owidth*dimensions[3]..' layer 3: number of features')
    fullyConnectedDim = owidth*owidth*dimensions[3]
    print(fullyConnectedDim..' final out layer')
    self.net:add(nn.Reshape(fullyConnectedDim))
    self.net:add(nn.Dropout(self.dropout[4]))
  else
    print(fullyConnectedDim..' final out layer')
    self.net:add(nn.Reshape(fullyConnectedDim))
    self.net:add(nn.Dropout(self.dropout[3]))
  end

--   linear to final
  self.net:add(nn.Linear(fullyConnectedDim, fullyConnectedDim))
  self.net:add(trans_layer[self.transfer]())
  self.net:add(nn.Linear(fullyConnectedDim, 2))  -- Should probably output to 1 neuron and then use something like logSigmoid instead of softmax

--   Final Layer
  self.net:add(nn.LogSoftMax())
  
  self:reset()
end
--]]

function convNet:reset()
  self.criterion = nn.ClassNLLCriterion()
  self.parameters, self.gradParameters = self.net:getParameters()
end


function convNet:n_batch()  -- WHAT IS THIS DOING???
  return 8
end

function convNet:trainStep(img,label)
  local output=self.net:forward(img)
  local err = self.criterion:forward(output,label)
  local df_dw = self.criterion:backward(output,label)
  self.net:backward(img,df_dw)
  return err, output
end

--[[
function convNet:augmentedTrainStep(img,label)
  local val = self:trainStep(img,label)
  local err = val['err']
  local output = {}
  table.insert(output, val['output'])
  val = self:trainStep(image.hflip(img),label)
  err = err + val['err']
  table.insert(output, val['output'])
  local tmp_img = img
  for i=1,3 do
    tmp_img = image.rotate(tmp_img,math.pi*0.5)
    val = self:trainStep(tmp_img,label)
    err = err + val['err']
    table.insert(output, val['output'])
    val = self:trainStep(image.hflip(tmp_img),label) 
    err = err + val['err']
    table.insert(output, val['output'])
  end
  return err, output
end
--]]


