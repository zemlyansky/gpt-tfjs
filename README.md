# GPT in TensorFlow.js (gpt-tfjs)
A minimal implementation of GPT architecture in TensorFlow.js more or less following the [minGPT](https://github.com/karpathy/minGPT) implementation in PyTorch. 

### Installation
```bash
npm i gpt-tfjs
```

### Example 
```javascript
const { GPTLMHeadModel } = require('gpt-tfjs')
const gpt = GPTLMHeadModel(config)
await gpt.train(train_dataset, {epochs: 10})
const inputs = [2, 2, 2, 1, 0, 1]
const idx = gpt.generate([inputs], 6)
```

### Testing
The testing script relies on minGPT to generate the test data. Before running tests run the following command (you'll need Pytorch installed):
```bash
python test_gen.py
```