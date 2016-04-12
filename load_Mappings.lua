
tds = require 'tds'
torch.setdefaulttensortype('torch.FloatTensor')
local indxToVocab = torch.load("indxToVocabB.t7")
local vocatToIndx = torch.load("vocabToIndxB.t7")
local dimension = 300
local file = "LearnedVectorsUniform.txt"
local h = tds.Hash()
for i=1,#indxToVocab do
  local word= indxToVocab[i]
  h[word] = torch.rand(300)

end
print("SAVING")
torch.save(file, h)

--print("LOADING")
--h = torch.load(file)

