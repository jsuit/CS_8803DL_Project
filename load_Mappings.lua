
tds = require 'tds'
torch.setdefaulttensortype('torch.FloatTensor')

local data = "06-comparativeDataSet.txt" 
pl = require 'pl.utils'
c = require 'csvigo'
local m = c.load({path=data,mode="large"})
local itoV = {}
local vtoI = {}
local count = 1
for i=1,#m do
	local line=m[i]
	assert(#line==1)
	local words = line[1]
	local words = pl.split(words)
	for j=1,#words do 
	        local word = words[j]
		--local words = pl.split(word)		
		word = word:gsub("%s+", "")	
		if word ~="" and word ~=" " then
			--print(word.. " ")
			if vtoI[word] == nil then
				vtoI[word] = count
				itoV[count] = word
				count = count + 1	
			end	
		end
	end
end
print("SAVING")
torch.save("IndxToVocabB.t7", itoV)
torch.save("vocabToIndxB.t7",vtoI)
print("DONE")

local indxToVocab = torch.load("IndxToVocabB.t7")
local vocatToIndx = torch.load("vocabToIndxB.t7")
local dimension = 300
local file = "LearnedVectorsUniform.txt"
local h = {}
for i=1,#indxToVocab do
  local word= indxToVocab[i]
  if word =="" or word == " " then

  else
	if word =="the" then print(word) end
  h[word] = torch.rand(300)
  end
end
print("SAVING")
torch.save(file, h)

--print("LOADING")
--h = torch.load(file)
