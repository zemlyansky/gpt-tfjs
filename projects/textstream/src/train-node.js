const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

const { encode, decode } = require('gpt-tokenizer/encoding/r50k_base')

const MLflow = require('./mlflow')
const client = new MLflow({endpoint: 'http://localhost:5000'})

const { createDatasetFromTextStreams, createDatasetFromFileList, vocabSize } = require('./create-dataset')
const { model, train, optimizers, utils } = require('../../../')

function createDatasetFromPaths (paths, blockSize) {
    paths = Array.isArray(paths) ? paths : [paths]
    function getStreams () {
        return paths.map(path => fs.createReadStream(path, { encoding: 'utf8', highWaterMark: 256 }))
    }
    return createDatasetFromTextStreams(getStreams, blockSize)
}

async function trainStream () {
    const modelType = 'gpt-nano'
    const blockSize = 64
    const batchSize = 4

    await client.createRun(0, `${modelType}-${tf.getBackend()}-${Date.now()}`)
    await client.logParam('modelType', modelType)
    await client.logParam('blockSize', blockSize)
    await client.logParam('batchSize', batchSize)
    
    const config = {
      // nLayer: 3,
      // nHead: 3,
      // nEmbd: 48,
      vocabSize: vocabSize,
      blockSize: blockSize,
      modelType: 'gpt-nano',
      dropout: 0.1,
      debug: false,
    }
    const gpt = model.GPTLMHeadModel(config)

    const filePath = path.join(__dirname, '../data/wiki.train.raw');
    const train_dataset = createDatasetFromPaths(filePath, blockSize)
    let timeNow = Date.now()

    const callback = async (m, loss, iteration) => {
      // Calculate time diff
      const time = Date.now()
      const timeDiff = time - timeNow
      timeNow = time
      await client.logMetric('time', timeDiff, iteration)

      // Log loss
      await client.logMetric('loss', loss, iteration)

      // Generate text
      const tokens = encode('Roses are red, violets are blue, ')
      const outputs = await model.generate(
          m,
          tokens, 
          { 'maxLength': 100, 'temperature': 0.7, 'doSample': false },
      )
      console.log('Output:', decode(outputs))
    }

    await gpt.train(train_dataset, {
      epochs: 10,
      maxIter: 100, 
      batchSize: batchSize, 
      verbose: true, 
      callbacks: [callback]
    })
}

trainStream().catch(console.error)