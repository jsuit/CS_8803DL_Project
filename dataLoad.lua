
--local vectors = require 'utils'.word2Vectors()
local pl = require 'pl.utils'


local dataLoad = torch.class('dataLoad')

function dataLoad:__init()

end
local insertAndCleanTables = function(dataTable,wordsT,targets,data)

   	table.insert(dataTable.targets, targets)
        table.insert(dataTable.words, wordsT)
        table.insert(dataTable.data, data)
	--targets is a tensor
        data={}
        wordsT = {}
	collectgarbage()
end

function dataLoad:getData()
  local vT = require 'utils'.getVocabMap()
  local indexToVocab = vT.indxToVocab

  assert(vT)
  local numWords = vT.numWords
  assert(numWords)
  print("Number of Words = " .. tostring(numWords))
  local numLines = vT.numLines
  local lines = vT.lines
  assert(lines)
  return vT
end

function dataLoad:getVectors(word2vec, style, dimension,vTable)
  print("Loading Vectors")
  assert(vTable)
  local vectors = require 'vectors'.initVectors()
  local indices = vTable.indxToVocab
  
  --for i=1, #indices do
--	assert(vectors[indices[i]])
--	vectors[indices[i]]= vectors[indices[i]]:cuda()
 -- end
  return vectors
end

local getSeq = function(maxSeqLen, line,vectors,vT)
  assert(maxSeqLen)
  assert(line)
  assert(vectors)
  local seqOfSequences = {}
  local words = line --pl.split(line)
  if words[1] == "" then
    table.remove(words,1)
  end
  local inputWordsTable = {}
  local tableOfWords = {}
  inputWordsTable = {}
  --loop over words
  local inputs = {}
  local targets = {}
  local targetsOfTargets = {}
  local numWords = #words
  --local targetsTensor = torch.Tensor()
  local vocabTable = vT
  assert(vT)
  local vocabToIndx = vocabTable.vocabToIndx
  local indexToVocab = vocabTable.indxToVocab
  assert(indexToVocab)
  assert(vocabToIndx)

  for i=1,numWords do
    if words[i] == "" then 

    else
    --keep adding to targets until we hit maxSeqLen, then add it to targetsOfTargets and clear out targets
    local target = torch.Tensor(1)
    if i+1 > numWords then
	target[1] = vocabToIndx["stop"]
	table.insert(targets, target:cuda()) 
    else
	assert(vocabToIndx[words[i+1]])
	target[1] = vocabToIndx[words[i+1]]
      table.insert(targets, target:cuda())
    end

    --inputs is a table with num entries == seqLen of size dimensions, where dimensions = v[word]:size(1)

    table.insert(inputs, vectors[words[i]]:cuda())
    table.insert(inputWordsTable, words[i])

    if i % maxSeqLen == 0 then
      table.insert(tableOfWords, inputWordsTable)
      table.insert(seqOfSequences,inputs)
    
      --local j = 0;
      --targetsTensor:apply(function() j=j+1; return targets[j] end)
      --targetsTensor:resize(#targets)
      table.insert(targetsOfTargets,targets)
      assert(#inputWordsTable == #inputs)
      assert(#inputs == #targets)
      --clear inputs
      inputs = {}
      targets ={}
      inputWordsTable ={}
    
      collectgarbage()
     end
    end
  end

  --sanity check
  if #targets > 0 then assert(#inputs >0) end

  if #inputs > 0 then
    assert(#targets > 0)
    --i % maxSeqLen == 0 will never be true
    table.insert(seqOfSequences,inputs)
    table.insert(targetsOfTargets,targets)
    table.insert(tableOfWords,inputWordsTable)

  end
  assert(#seqOfSequences == #targetsOfTargets)
  assert(#targetsOfTargets == #tableOfWords)
  return {seqOfSequences, targetsOfTargets,tableOfWords}
end


function dataLoad:getNextSequences(maxSeqLen,lines,vectors,vT)
	--assert(maxSeqLen)
	--assert(line)
	--assert(vectors)
	--assert(vT)
  --seqofSequenes == table of tables of 
  	seqSeqTargetsTable ={}
	for i=1,#lines do
		seqSeqTargetsTable[i] = {}
		tempTable = getSeq(maxSeqLen,lines[i], vectors,vT)
		seqSeqTargetsTable[i].data=tempTable[1]
		seqSeqTargetsTable[i].targets=tempTable[2]
		seqSeqTargetsTable[i].words=tempTable[3]
	end
  	--local seqSeqTargetsTable = getSeq(maxSeqLen,line, vectors,vT)
  	return seqSeqTargetsTable 
	--{data = seqSeqTargetsTable[1], targets=seqSeqTargetsTable[2],words = seqSeqTargetsTable[3]}


end

function dataLoad:sample(batchsize,seqLen, lines, vectors,hiddensize,vocabToIndx)
	local seqLen = math.random(4,seqLen)
	local vocabToIndx = vocabToIndx
	local dataTable= {}
	dataTable["data"] = {}
	torch.setdefaulttensortype("torch.CudaTensor")
	dataTable["targets"] = {}
	dataTable["words"] ={}
		local words = lines[math.random(1,#lines)]
		if words[1] == "" then 
                	table.remove(words,1)
        	end
		while #words < seqLen do
			words = lines[math.random(1,#lines)]
		end
		local start = math.random(1,#words - seqLen + 1)
		local vTensor = torch.Tensor(hiddensize)
		local targetTensor = torch.Tensor(seqLen)
		if start+ seqLen-1 == #words then
			table.insert(words,"stop")
		end
		for i=1,seqLen do 
			local word = words[start]
			table.insert(dataTable.data, vectors[word])
			targetTensor[i] = vocabToIndx[word]
			table.insert(dataTable.targets, targetTensor[i])
			table.insert(dataTable.words, word)
			start = start+1
		end
	
	return dataTable
end

