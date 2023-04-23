// Saving a modle with JEST throws Unsupported TypedArray subtype: Float32Array
// https://stackoverflow.com/questions/57452981/tensorflowjs-save-model-throws-error-unsupported-typedarray-subtype-float32arr
const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')
const { 
    CausalSelfAttention,
    CausalSelfAttentionMixed,
} = require("./src/model")

const tests = require('./test.json')
const input = tf.tensor(tests.att_grads.inputs[0])
const output = tf.tensor([[[ -10, 8, -6, -4 ], [ 20, 10, -20, 4 ], [ 0, 0, -20, 40 ]]])
const config = tests.att_grads.configs[0]
config.dropout = 0

console.log(config)

const createAttentionModel = (createAttention) => {
    const x = tf.input({shape: [input.shape[1], input.shape[2]]})
    const y = createAttention(Object.assign({name: 'attn_1'}, config)).apply(x)
    const z = createAttention(Object.assign({name: 'attn_2'}, config)).apply(y)
    const model = tf.model(Object.assign({ inputs: x, outputs: z }, config))
    model.getWeights().forEach(w => { w.assign(tf.ones(w.shape)) })
    return model
}

const model1 = createAttentionModel(config => CausalSelfAttention(config))
// exit
const model2 = createAttentionModel(config => CausalSelfAttentionMixed(config))

const loss = (preds, labels) => tf.losses.meanSquaredError(preds, labels)
const optimizer1 = tf.train.adam(0.1)
const optimizer2 = tf.train.adam(0.1)
const train = (model, optimizer) => {
    const optFunc = () => {
        const preds = model.apply(input, {training: true}) // training: true is required for dropout
        return loss(preds, output)
    }
    tf.tidy(() => {
        var {values, grads} = optimizer.computeGradients(optFunc)
        optimizer.applyGradients(grads)
    })
    // console.log('train', model.getWeights().forEach(w => { console.log(w.arraySync()) }))
}
for (let i = 0; i < 100; i++) {
    train(model1, optimizer1)
    train(model2, optimizer2)
}
// Get weights
const weights1 = model1.getWeights()
const weights2 = model2.getWeights()
// Compare weights
const mses_1_2 = []
weights1.forEach((w1, i) => {
    const w2 = weights2[i]
    console.log('mse between w1 and w2', w1.name, tf.losses.meanSquaredError(w1, w2).arraySync())
    const mse = tf.losses.meanSquaredError(w1, w2).arraySync()
    mses_1_2.push(mse)
})  
console.log('mse between m1 and m2', tf.mean(mses_1_2).arraySync())

// Save and load
// Check if the directory exists
if (!fs.existsSync('temp')) {
    // Create the directory
    fs.mkdirSync('temp')
}
const path = process.cwd() + '/temp/test_model'

;(async () => {
    await model1.save(`file://${path}_1`)
    await model2.save(`file://${path}_2`)
    const model1Loaded = await tf.loadLayersModel(`file://${path}_1/model.json`)
    const model2Loaded = await tf.loadLayersModel(`file://${path}_2/model.json`)
    
    // Get weights
    const weights1Loaded = model1Loaded.getWeights()
    const weights2Loaded = model2Loaded.getWeights()

    // Compare weights
    var mses = []
    weights1Loaded.forEach((w1l, i) => {
        const w1 = weights1[i]
        const mse = tf.losses.meanSquaredError(w1l, w1).arraySync()
        mses.push(mse)
    })
    weights2Loaded.forEach((w2l, i) => {
        const w2 = weights2[i]
        const mse = tf.losses.meanSquaredError(w2l, w2).arraySync()
        mses.push(mse)
    })
    console.log('mse between saved and loaded', tf.mean(mses).arraySync())
    
    for (let i = 0; i < 10; i++) {
        console.log(model1Loaded.predict(input).arraySync()[0][0][0])
    }
})()