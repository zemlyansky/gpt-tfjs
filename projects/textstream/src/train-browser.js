// const tf = require('@tensorflow/tfjs')
const { encode, decode } = require('gpt-tokenizer/encoding/r50k_base')

const MLflow = require('./mlflow')
const client = new MLflow({endpoint: 'http://localhost:5000'})

const { createDatasetFromTextStreams, createDatasetFromFileList, vocabSize } = require('./create-dataset')
const { model, train, optimizers, utils } = require('../../../')

async function sleep (t) {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, t);
  });
}

function countParams(model) {
  return model.getWeights()
    .filter(weight => weight.trainable)
    .filter(weight => weight.name !== 'lm_head/kernel')
    .reduce((total, weight) => total + weight.size, 0);
}

async function handle() {
    // const files = Array.from(this.files)
    // const streams = files.map(file => new ReadStream(file, { chunkSize: 4096 }))

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

    const info = document.getElementById('info')
    for (let i = 0; i < this.files.length; i++) {
      const file = this.files[i]
      info.innerHTML += '<br>File: ' + file.name + ' ' + file.size + ' bytes'
    }

    const gpt = model.GPTLMHeadModel(config)
    window.gpt = gpt
    console.log('Model config:', config)
    console.log('Model params:', countParams(gpt.model))
    
    info.innerHTML += '<br>Model config: ' + JSON.stringify(config)
    info.innerHTML += '<br>Model params: ' + countParams(gpt.model)

    const train_dataset = createDatasetFromFileList(this.files, config.blockSize)
    
    // const start = performance.now()
    let calls = 0
    let chart = null
    let g = null
    let timeNow = Date.now()
    const losses = []
    const ctx = document.getElementById('plot')
    const callback = async (m, loss, iteration) => {
      // Log each 10th iteration
      // if (iteration % 10 !== 0) {
      //   return
      // }

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

      losses.push(loss)
      if (chart) {
        chart.data.datasets[0].data = losses
        chart.data.labels = losses.map((_, i) => i)
        chart.update()
      } else {
        chart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: losses.map((_, i) => i),
            datasets: [{
              label: 'Loss',
              data: losses,
              backgroundColor: 'rgba(255, 99, 132, 0.2)',
              borderColor: 'rgb(255, 99, 132)',
              borderWidth: 1
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
          }
        })
      }

      const output = document.getElementById('output')
      output.innerHTML = '<br>' + decode(outputs[0])
      calls++
    } 

    await gpt.train(train_dataset, {
      epochs: 10,
      maxIter: 100, 
      batchSize: batchSize, 
      verbose: true, 
      callbacks: [callback]
    })
}

const inputElement = document.getElementById("input")
inputElement.addEventListener("change", handle, false)

