return local f = function(dataTable)
  local seqOfSeq = dataTable.data
  local seqOfTargets = dataTable.targets
  local wordTable
  if backpropToWord then
    wordTable = dataTable.words
  end

  --for each line get seq of sequences. #seqOfSeq tells us how many sequences of maxSeqLen we could make
  -- from jth line.
  -- if the last seqOfSeq might be less than maxSeqLen we could make
  -- if #words in lines[j] < maxSeqLen, then #seqOfSeq =1 and seqOfSeq[1] == #words in lines[j]
--	print("seqOfSeq= " .. tostring(#seqOfSeq)) 

  for k =1, #seqOfSeq do
    local eval = function(x)
      collectgarbage()
      grad_params:zero()
      local data = seqOfSeq[k]
      local output = model:forward(data)
      local err = criterion(output,seqOfTargets[k])
      model:backward(data, criterion:backward(output, seqOfTargets[k]))
      grad_params:clamp(-grad_clip, grad_clip)
      --print(err)

      return err, grad_params
    end

    _, E = optimMethod(eval,params, adam_params)
    if adam_params.t ~= nil and adam_params.t % 2 == 0 then
      print("Epoch " .. tostring(i) .. " iteration " .. tostring(adam_params.t))
      print("Error " .. tostring(E[1]))
    end
    curError = curError+ E[1]
    --if E[1] < 50 then
    --require 'mobdebug'.start()
    --end
    trainLogger:add{['% CE (train set)']=E[1]}
    trainLogger:style{['% CE (train set)'] = '-'}
    --trainLogger:plot()
    if backpropToWord then
      collectgarbage()
      local gradTable = model.modules[1].gradInput
      assert(wordTable)
      local words= wordTable[k]

      assert(#words == #gradTable)
      for i=1, #gradTable do
        --local fevalWord = function() return 0,model.modules[1].gradInput[i]:float()end  
        local vCuda = vectors[words[i]]
        optimMethod(function(x) return 0,gradTable[i] end,vCuda,wordOptim)	  
        --vCuda:add(-1*learningRate,model.modules[1].gradInput[i])

        vCuda:div(vCuda:norm())
        vectors[words[i]] = vCuda
        --vectors[words[i]]:div(vectors[words[i]]:norm())
        if vocabToIndx[words[i]] ~= nil then
          if seenVocab[words[i]] == nil then seenVocab[words[i]] = 1
          else seenVocab[words[i]] = seenVocab[words[i]] + 1
          end
        end
        if #seenVocab == numVocab then
          csvigo.save({path="LearnedVectors_" .. tostring(i) .. "_" .. tostring(optimState.t) .. ".csv",data=vectors})
          seenVocab = {} 
          collectgarbage()
        end
        if #seenVocab == .5*numVocab then
          print("SEEN 1/2 of all VOCAB")
        end
        if #seenVocab	== .25*numVocab then
          print("SEEN 1/4 of all VOCAB")
        end 


      end

    end
  end
end