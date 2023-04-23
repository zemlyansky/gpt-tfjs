# GPT in TensorFlow.js (gpt-tfjs)
A minimal implementation of GPT architecture in TensorFlow.js more or less following the [minGPT](https://github.com/karpathy/minGPT) implementation in PyTorch. 

### Installation
```bash
npm i gpt-tfjs
```

### Example 
```javascript
const tf = require('@tensorflow/tfjs') // or @tensorflow/tfjs-node or @tensorflow/tfjs-node-gpu
const { GPTLMHeadModel } = require('gpt-tfjs').model
const config = {
  nLayer: 3,
  nHead: 3,
  nEmbd: 48,
  vocabSize: 3,
  blockSize: 11,
  dropout: 0.1
}
(async () => {
  const gpt = GPTLMHeadModel(config)
  await gpt.train(trainDataset, {epochs: 10, verbose: true})
  const inputs = [2, 2, 2, 1, 0, 1]
  const idx = gpt.generate([inputs], 6)
  console.log(idx.arraySync()[0].slice(6)) // [0, 1, 1, 2, 2, 2]
})()
```
Where `trainDataset` is a tensorflow dataset, for example: [https://github.com/zemlyansky/gpt-tfjs/blob/main/projects/sort/sort.js](https://github.com/zemlyansky/gpt-tfjs/blob/main/projects/sort/sort.js)

### Testing
The testing script relies on minGPT to generate the test data. Before running tests run the following command (you'll need Pytorch installed):
```bash
python test_gen.py
```