require 'rnn'
require 'optim'
require 'dpnn'
torch.setdefaulttensortype("torch.FloatTensor")
version = 1.2 -- refactored numerical gradient test into unit tests. Added training loop
local word2vec 
local style = "uniform"
local hiddenSize =300
torch.manualSeed(1)
require 'dataLoad'
local dataLoader = dataLoad()
local dataTableChar = dataLoader:getDataAndCharMapping()

local charToIndx = dataTableChar[2]
local indxToChar = dataTableChar[3]
local dataTable = dataTableChar[1]
local vocabToIndx = dataTable.vocabToIndx
local indxToVocab = dataTable.indxToVocab
local lines = dataTable.lines

local opt = {}
opt.learningRate = .15
opt.seqLen =  10 -- length of the encoded sequence
opt.hiddenSize = hiddenSize

opt.vocabSize = #dataTable.indxToVocab
local batchSize=1
--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
local function forwardConnect(encLSTM, decLSTM)
  --require 'mobdebug'.start()
  local size = #encLSTM.outputs
  decLSTM.userPrevOutput = nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[size])
  decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[size])
end
--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
local function backwardConnect(encLSTM, decLSTM)
  encLSTM.userNextGradCell = nn.rnn.recursiveCopy(encLSTM.userNextGradCell, decLSTM.userGradPrevCell)
  encLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encLSTM.gradPrevOutput, decLSTM.userGradPrevOutput)
end
--Encoder

local enc = nn.Sequential()
local lt1 = nn.LookupTable(#indxToChar, opt.hiddenSize)
lt1.maxOutNorm =false
enc:add(lt1)
enc:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
local encLSTM = (nn.LSTM(opt.hiddenSize, opt.hiddenSize))
encLSTM.usenngraph=true
enc:add(nn.Sequencer(encLSTM:maskZero(1)))
enc:add(nn.SelectTable(-1))
enc = require('weight_init')(enc, 'xavier')
--Decoder
local dec = nn.Sequential()
local lt = nn.LookupTable(#indxToVocab, opt.hiddenSize)
lt.maxOutNorm = false
dec:add(lt)
dec:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
local decLSTM = nn.LSTM(opt.hiddenSize, opt.hiddenSize)
--decLSTM.usenngraph=true
dec:add(nn.Sequencer(decLSTM:maskZero(1)))
local linear = nn.Linear(opt.hiddenSize, opt.vocabSize)

dec:add(nn.Sequencer(linear))
dec:add(nn.Sequencer(nn.LogSoftMax()))
--enc:remember('both')
dec = require('weight_init')(dec, 'xavier')
dec:remember('both')
--enc = nn.MaskZero(enc,1)
--dec = nn.MaskZero(dec,1)
local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

local trainLogger = optim.Logger('train.log')
trainLogger:style{['NLL'] = '-'}
local accuracy = optim.Logger("accuracy.log")

local curEpoch = 1
--p1,g1 = enc:getParameters()
--p2,g2 = dec:getParameters()

local maxEpoch = 200
for i=curEpoch,maxEpoch do


  print("EPOCH: " .. tostring(i))
  local timer = torch.Timer()
  --for each epoch
  torch.setdefaulttensortype('torch.FloatTensor')
  local indices = torch.randperm(#lines)
  enc:training()
  dec:training()
  for j=1,indices:size(1) do
    print("line " ..j)

    local seqs = dataLoader:getNextSeqOfChars(batchSize, opt.seqLen,lines[indices[j]],vocabToIndx,charToIndx,indxToChar)

    local decInSeq = seqs.decInSeq
    local decOutSeq = seqs.decOutSeq
    local stop = 8
    enc:forget()
    --if j % 4 == 0 then
      --dec:forget()
    --end

    for k=1,decInSeq:size(1),1 do
      print(k)

      if k+8 <= decInSeq:size(1) then
        stop = 8
      else
        stop=decInSeq:size(1)-k+1
      end
      local decInSeqT= nn.NarrowTable(k,stop):forward(decInSeq)
      local decOutSeqT = nn.NarrowTable(k ,stop):forward(decOutSeq)
      assert(decInSeqT)
      assert(decOutSeqT)
      --print(decInSeqT)
      --print(decOutSeqT)
      local chars = {}
      for w=1,#decInSeqT do
        --each word
        local word = indxToVocab[decInSeqT[w]]
        for ch=1,#word do           
          if word:sub(ch,ch) == nil then
            print("word == nil")
          end
          table.insert(chars,torch.Tensor({charToIndx[word:sub(ch,ch)]}))
        end
      end

      local encInSeq = nn.JoinTable(1):forward(chars)
      local tensor_decInSeq = torch.Tensor(decInSeqT)
      enc:zeroGradParameters()
      dec:zeroGradParameters()


      local encOut = enc:forward(encInSeq)
      collectgarbage()
      --local dfdx =  function(x)
      forwardConnect(encLSTM, decLSTM)
      --require 'mobdebug'.start()

      local decOut = dec:forward(tensor_decInSeq)
      --print(decOutSeqT)

      --for t=1,#decOut do
      --require 'mobdebug'.start()
      --print(decOut[t]:mean())
      --end
      --print(confusion)


      local err = criterion:forward(decOut, decOutSeqT)
      print(err)
      trainLogger:style{['NLL'] = '-'}
      trainLogger:add{['NLL'] =  err/#decOutSeqT}
      trainLogger:plot()
      collectgarbage()
      local gradOutput = criterion:backward(decOut, decOutSeqT)
      dec:backward(tensor_decInSeq, gradOutput)
      collectgarbage()

      backwardConnect(encLSTM, decLSTM)
      local z = torch.Tensor(encOut):zero()
      --encLSTM.gradPrevOutput
      encOut=nil
      collectgarbage()
      enc:backward(encInSeq,z)
      enc:updateGradParameters(.95,.05, true)
      dec:updateGradParameters(.95,.05, true)

      dec:updateParameters(opt.learningRate)
      enc:updateParameters(opt.learningRate)
      dec:maxParamNorm(2)
      enc:maxParamNorm(2)
      dec:gradParamClip(5)
      enc:gradParamClip(5)
      
      
      --optim.adam(dfdx,p2, {learningRate=.001})
      --optim.adam(function(x) return 0,g1 end,p1, {learningRate=.001})
      --print(p1:mean())
      --print(p2:mean())
      --print(g1:mean())
      --print(g2:mean())
      collectgarbage()




      collectgarbage()
    end
  end
end


