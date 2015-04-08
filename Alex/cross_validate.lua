print '==> defining test procedure'
require 'math'

-- test function
function cross_validate(epoch,netObject,cvData)  -- epoch counts number of times through training data
  
  local localNet = netObject:getNet()
  -- averaged param use?
  if average then
    cachedparams = parameters:clone()
    parameters:copy(average)
  end
  
  -- test over test data
  print('==> testing on test set:')
  local time = sys.clock()
  
  -- This matrix records the current confusion across classes (2x2 for surveyor)
  local confusion = optim.ConfusionMatrix({'0','1'})
  

  local points = torch.Tensor(#cvData.images)
  for test_example = 1,#cvData.images do
    -- disp progress
    xlua.progress(test_example, #cvData.images)
    
    -- test sample
    local pred = localNet:forward(cvData.images[test_example])
    
    --local max_val = torch.Tensor()
    --local max_index = torch.LongTensor()
    --pred.max(max_val,max_index,pred,1)
    
    --print('Prediction: ' .. species[ max_index[1] ])
    --if species[ max_index[1] ] ~= plankton_targets_cv[test_example] then
    --  misclassify[ plankton_targets_cv[test_example] ] = misclassify[ plankton_targets_cv[test_example] ] or {}
    --  misclassify[ plankton_targets_cv[test_example] ][ species[ max_index[1] ] ]
    --  = misclassify[ plankton_targets_cv[test_example] ][ species[ max_index[1] ] ] or {}
    --  table.insert(misclassify[ plankton_targets_cv[test_example] ][ species[ max_index[1] ] ],
    --    plankton_paths_cv[ test_example ] )
    --end
    
    --points[test_example] = - pred[ plankton_ids[ plankton_targets_cv[test_example] ] ]
    
    confusion:add(pred, cvData.labels[test_example])
  end
  points:div(#cvData.images)
  
  -- timing
  time = sys.clock() - time
  if #cvData.images ~= 0 then time = time / #cvData.images end
  print("==> time to test 1 sample = " .. (time*1000) .. 'ms')
  
  -- print confusion matrix
  print(confusion)
  confusion:zero()
	
  --local score = torch.sum(points)
  --print('Score: ' .. score)
  --torch.save('../logprob/points_'..epoch..'.dat', torch.Tensor(points) )

  -- update log/plot
  --testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  --if opt.plot then
  --	testLogger:style{['% mean class accuracy (test set)'] = '-'}
  --	testLogger:plot()
  --end
  
  -- averaged param use?
  if average then
    -- restore parameters
    parameters:copy(cachedparams)
  end
  
  --return(points)
end
