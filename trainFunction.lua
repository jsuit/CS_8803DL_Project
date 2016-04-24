local grad_clip = 2

local adam_params = {
  learningRate = 2e-3,
  alpha=.95
}
local wordOptim = {
  learningRate = 1e-4,
  learningRateDecay = 1e-5,
  weightDecay =1e-5,
  momentum = .95
}
local saving= function(table,filename)
	for key,vec in pairs(table) do
		table[key] = vec:float()
	end
	torch.save(filename,table)
	for key,vec in pairs(table) do
		table[key] = vec:cuda()
	end
end

local backpropToWord=true  
local seenVocab={}
local seenVocabCount = 0
local f = function(Epoch,dataTable,optimMethod,model,curError,vectors,vocabToIndx,wordRepVocab,wordRepCount)
  
  local seqOfSeq = dataTable.data
  assert(wordRepVocab)
  local seqOfTargets = dataTable.targets
  local wordTable
  if backpropToWord then
    wordTable = dataTable.words
  end
    local totalError = 0
    model:forget()
    for k=1,#seqOfSeq do
  --for each line get seq of sequences. #seqOfSeq tells us how many sequences of maxSeqLen we could make
  -- from jth line.
  -- if the last seqOfSeq might be less than maxSeqLen we could make
  -- if #words in lines[j] < maxSeqLen, then #seqOfSeq =1 and seqOfSeq[1] == #words in lines[j]
--	print("seqOfSeq= " .. tostring(#seqOfSeq)) 

    local eval = function(x)
      collectgarbage()
      grad_params:zero()
      local data = seqOfSeq[k]
      local output = model:forward(data)
      local err = criterion(output,seqOfTargets[k])
      model:backward(data, criterion:backward(output, seqOfTargets[k]))
      --print(err)
      print('gp = ' .. grad_params:mean())
      print('p = ' .. params:mean())	
      return err, grad_params
    end
    _, E = optimMethod(eval,params, adam_params)
    grad_params:clamp(-grad_clip, grad_clip)
    local normError =  E[1]/#(seqOfSeq[k]) 
    print(string.format("Norm Error = %d",normError))
    totalError = totalError + E[1]
    --if adam_params.t == nil then adam_params.t =0 end
    --adam_params.t = adam_params.t+1
    if adam_params.t ~= nil and adam_params.t % 2 == 0 then
      print("Epoch " .. tostring(Epoch) .. " iteration " .. tostring(adam_params.t))
      print("Error " .. tostring(E[1]))
      print("NormError ",tostring(normError))
      print("Seq length = " .. tostring(#(seqOfSeq[k])))
    end
    curError = curError+ E[1]
    --if E[1] < 50 then
    --require 'mobdebug'.start()
    --end
    trainLogger:add{['% CE (train set)']=normError}
    trainLogger:style{['% CE (train set)'] = '-'}
    --trainLogger:plot()
  assert(wordRepVocab)
    if backpropToWord then
      collectgarbage()
      local gradTable = model.modules[1].gradInput
      --print("SeqLen = " .. #wordTable[k])
      --print(#wordTable[k], #gradTable)
      assert(#wordTable[k] == #gradTable)
      local  words= wordTable[k]
      for i=1,#gradTable do
	assert(wordRepVocab)
        --local fevalWord = function() return 0,model.modules[1].gradInput[i]:float()end  
        local vCuda = vectors[words[i]]
        optimMethod(function(x) return 0,gradTable[i] end,vCuda,wordOptim)	  
        --vCuda:add(-1*learningRate,model.modules[1].gradInput[i])
	
        vCuda:div(vCuda:norm())
        vectors[words[i]] = vCuda
        if wordRepVocab[words[i]] ~= nil then 
		if seenVocab[words[i]] == nil then
			seenVocab[words[i]] = 1
			seenVocabCount = seenVocabCount + 1
		else
			seenVocab[words[i]] = seenVocab[words[i]] + 1
		end
		print(string.format("Seen %s %d times",words[i],seenVocab[words[i]]))
		print("Seen this many vocab words: " .. tostring(seenVocabCount))
	end
        --vectors[words[i]]:div(vectors[words[i]]:norm())
        if seenVocabCount == wordRepCount then
	  local dateTable = os.date("*t")
	  local fileName = "LearnedVectors_" .. 
	  dateTable.month .. "_" .. dateTable.day .. "_" .. dateTable.hour
	  .. "_".. "Iter_"..adam_params.t ..".csv"
	 print("SAVING VECTORS") 
	 saving(vectors,fileName)
          
	  seenVocab = {} 
	  seenVocabCount = 0
          collectgarbage()
	  collectgarbage()
        end
        if  seenVocabCount == .5*wordRepCount then
          print("SEEN 1/2 of all VOCAB")
        end
        if seenVocabCount == .25*wordRepCount then
          print("SEEN 1/4 of all VOCAB")
        end 
       end

      end
    end
end
return f
