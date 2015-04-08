
require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'
require 'xlua'  -- progress bar
require 'optim'  -- confusion matrix; gradient decent optimization
require 'paths'  -- read OS directory structure

dofile 'convnet.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-trainingdir', '../train', 'location of training directory')
cmd:option('-testingdir', '../test', 'location of testing directory')
cmd:option('-misclassifieddir', '../misclassdata_path', 'location of misclassified directory')
cmd:option('-preprocesseddir', '../preprocessed', 'location of preprocessed directory')
cmd:option('-netDatadir', '../NNsave', 'location of neural net data directory')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-plot', false, 'live plot')
cmd:option('-trainOn', 80, 'percent of training data to use for training; the rest if used for cross-validation')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-1, 'learning rate at t=0')  -- could make this variable.  Start big and decay.
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-sgdUntil',2,'epoch at which to switch over to different optimization')
cmd:option('-weightDecay', 0.0, 'weight decay (SGD only)')
cmd:option('-transfer', 'ReLU', 'transfer function: Tanh | ReLU | Sigmoid')
cmd:option('-dropout', '0,0.2,0.2', 'fraction of connections to drop: comma seperated numbers in the range 0 to 1')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 4, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-maxEpoch', 8, 'maximum number of epochs during training')  -- set to -1 for unlimited epochs
cmd:option('-loadNet', '', 'load from previous opt')
cmd:option('-learningRateScale', 0.5, 'factor to reduce the learning rate if the score is increasing')
cmd:text()
opt = cmd:parse(arg or {})

local dropout = {}
for match in opt.dropout:gmatch("([%d%.%+%-]+),?") do
  table.insert(dropout,tonumber(match))
end
opt.dropout = dropout

preprocessed_images = torch.load('half_disks.dat')  -- loads array preprocessed_images, cotaining buffered tensors of size theta,r = 20,16
solution_map = torch.load('solution.dat')

-- seperate training from cross-validation sets.
-- The approach here is somewhat naive because images in different sets have a lot of overlap in the full image.  Could be improved.
local trainData, cvData = {}, {}
trainData.images, trainData.labels, cvData.images, cvData.labels = {}, {}, {}, {}
local shuffle = torch.randperm(#preprocessed_images)
for i = 1, math.floor(#preprocessed_images * opt.trainOn/100) do
	table.insert(trainData.images, preprocessed_images[ shuffle[i] ])
	table.insert(trainData.labels, solution_map[ shuffle[i] ])
end
for i = math.floor(#preprocessed_images * opt.trainOn/100) + 1, #preprocessed_images do
	table.insert(cvData.images, preprocessed_images[ shuffle[i] ])
	table.insert(cvData.labels, solution_map[ shuffle[i] ])
end








myNet = convNet(opt)
-- this is the definition of the net. we will rewrite the command line arguments to be able to define this and some options at runtime.
-- then we can write a bash script to scan input space and find the best settings
-- myNet:build({1,64,128,64,#species}, 2, 2)

if opt.loadNet~='' then
  print( 'Loading '.. opt.netDatadir..'/'..opt.loadNet..'.dat')
  myNet.net = torch.load(opt.netDatadir..'/'..opt.loadNet..'.dat')
  myNet:reset()
else
  print('Building a new net to train')
  --myNet:build({32,64}, {3,2,2}, {1,1,1}, {2,2,1})  -- dimensions, filter dim (should be just azimuth dim; radial dim = radius), stride, pools
  myNet:build()
end

--output = myNet:trainStep(torch.Tensor(1,20,16),1)
--print(output.output, output.err)

dofile 'config_optimizer.lua'  -- optim.sgd, optim.asgd, optim.lbfgs, optim.cg 
dofile 'train.lua'  -- train convnet
dofile 'cross_validate.lua'  -- test convnet with cross-validation data
--dofile 'save_nets.lua'

--netSaver = save_nets(opt)
--netSaver:prepNNdirs()

--local points = {}
--local time_stamps = {}
--local scores = {}
--table.insert(scores,1e9)

for epoch = 1, opt.maxEpoch do
  train(epoch,myNet,trainData)
  cross_validate(epoch,myNet,cvData)
  --local points_distr = test(epoch,myNet)
  --local score = torch.sum(points_distr)
  --if epoch ~= 1 and score>scores[epoch-1] then
  --  print('Score went up! Reducing learning rate')
  --  optimState.learningRate = optimState.learningRate * opt.learningRateScale
  --end
  --netSaver:saveNN(epoch,myNet:getNet())
  --table.insert(scores,score)
  --table.insert(points,points_distr)
  --table.insert(time_stamps,os.date())
end

-- copy net with minimal cross-validation score to NN.dat
--local min_val = torch.Tensor()
--local min_index = torch.LongTensor()
--torch.min(min_val,min_index,torch.Tensor(scores),1)
--netSaver:saveBestNet(min_index[1],time_stamps[ min_index[1] ],scores[ min_index[1] ], points[ min_index[1] ], opt)
