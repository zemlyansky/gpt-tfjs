// Saving a modle with JEST throws Unsupported TypedArray subtype: Float32Array
// https://stackoverflow.com/questions/57452981/tensorflowjs-save-model-throws-error-unsupported-typedarray-subtype-float32arr
const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')
const { model, train, optimizers, utils } = require('../../')

const config = {
  nLayer: 3,
  nHead: 3,
  nEmbd: 48,
  vocabSize: 3,
  blockSize: 11,
  dropout: 0.1,
  debug: false
}

function createDataset(split='train', length=6, num_digits=3) {
    const vocabSize = num_digits
    const hashCode = s => s.split('').reduce((a,b) => (((a << 5) - a) + b.charCodeAt(0))|0, 0)

    function* dataGenerator () {
        let index = 0;
        while (index < 128) {
            let inp
            while (true) {
                inp = (new Array(length)).fill(0)
                    .map((v, i) => Math.floor(Math.random() * vocabSize))

                if (Math.random() < 0.5) {
                    const inpUnique = new Set(inp)
                    if (inpUnique.size > Math.floor(length / 2)) {
                        continue
                    }
                }
                const h = hashCode(inp.join(''))
                const inpSplit = h % 4 === 0 ? 'test' : 'train'
                if (inpSplit == split) {
                    break
                }
            }
            let sol = inp.slice()
            sol.sort()
            let cat = inp.concat(sol)
            let x = cat.slice(0, -1)
            let y = cat.slice(1).map((v, i) => i < length - 1 ? -1 : v)

            index++;
            yield {x, y};
        }
    }

    return tf.data.generator(dataGenerator)
        .map(v => ({
            x: tf.cast(v.x, 'int32'), 
            y: tf.oneHot(tf.cast(v.y, 'int32'), vocabSize)
        }))
}

async function trainSort () {
    const train_dataset = createDataset('train')
    const gpt = model.GPTLMHeadModel(config)
    const start = performance.now()
    await gpt.train(train_dataset, {epochs: 10}) // Expect this API to change
    const time = performance.now() - start
    console.log(`Training took ${time}ms`)
    var idx = gpt.generate([[2, 2, 2, 1, 0, 1]], 6)
    console.log('Pred:', idx.arraySync()[0].slice(6))
    return gpt
}

// Check if it's running directly, in a browser or loaded as a module with require
if ((typeof window !== 'undefined') || (require.main === module)) {
    trainSort()
} else {
    module.exports = {
        createDataset
    }
}
