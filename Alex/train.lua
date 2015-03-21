
function table_shuffle(table_of_tables)
	local shuffled_table_of_tables = {}
	for key,value in pairs(table_of_tables) do
		shuffled_indices = shuffled_indices or torch.randperm(#value)
		shuffled_table_of_tables[key] = {}
		for i = 1, #value do
			table.insert(shuffled_table_of_tables[key], value[ shuffled_indices[i] ])
		end
	end
	return shuffled_table_of_tables
end


function train(epoch,netObject,trainData)  -- epoch counts number of times through training data
  -- local vars
  local time = sys.clock()
  
  -- This matrix records the current confusion across classes (2x2 for surveyor)
  local confusion = optim.ConfusionMatrix({'0','1'})
  
  -- do one epoch
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  trainData = table_shuffle(trainData)

  for batch_start_example = 1, #trainData.images, opt.batchSize do
    -- disp progress
    xlua.progress(batch_start_example, #trainData.images)
    
    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new net.parameters
      if x ~= netObject.parameters then
        netObject.parameters:copy(x)
      end
      --print(net.parameters)
      
      -- reset gradients
      netObject.gradParameters:zero()
      
      -- f is the average of all net.criterions
      local f = 0
      
      -- evaluate function for complete mini batch
      local batch_size = 0  -- keeps track of actual batch size (since last batch may be smaller)
      for batch_example = batch_start_example,math.min(batch_start_example + opt.batchSize - 1, #trainData.images) do
        batch_size = batch_size + netObject:n_batch()
        local err, output = netObject:trainStep(trainData.images[batch_example], trainData.labels[batch_example])
        f = f + err
		
        -- update confusion
        confusion:add(output, trainData.labels[batch_example])
      end
      
      -- normalize gradients and f(X)
      netObject.gradParameters:div(batch_size)
      f = f / batch_size
      
      -- return f and df/dX
      return f,netObject.gradParameters
    end
  
  -- optimize on current mini-batch
  if optimMethod == optim.asgd then
    _,_,average = optimMethod(feval, netObject.parameters, optimState)
  else
    optimMethod(feval, netObject.parameters, optimState)
  end
end

-- print confusion matrix
print(confusion)
confusion:zero()

-- update logger/plot
--trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
--if opt.plot then
--	trainLogger:style{['% mean class accuracy (train set)'] = '-'}
--	trainLogger:plot()
--end

-- save/log current net
--local filename = paths.concat(opt.save, 'net.net')
--os.execute('mkdir -p ' .. sys.dirname(filename))
--print('==> saving net to '..filename)
--torch.save(filename, net)


confusion:zero()

end
