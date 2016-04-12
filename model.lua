require 'dpnn'
require 'rnn'
local dataLoader = require 'dataLoad'
local grad_clip =5
local word2vec = false
local style = "random"

local batchSize = 1

local hiddenSize = 300
local dataTable = dataLoader.getData()
assert(dataTable)
local indxToVocab = dataTable.indxToVocab
local nIndex = #indxToVocab
model = nn.Sequential()
model:add(nn.Sequencer(nn.LookupTable(hiddenSize, hiddenSize)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize, nIndex)))
model:add(nn.Sequencer(nn.LogSoftMax()))

--x,dx = model:getParameters()
collectgarbage()
collectgarbage()
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
return model