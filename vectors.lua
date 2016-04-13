local vectors = {}

function vectors.initVectors(file)

  if file == nil then
    return torch.load("LearnedVectorsUniform.txt")
  else
    return torch.load(file)
  end
end
return vectors