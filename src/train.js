const tf = require('@tensorflow/tfjs')
const { AdamW, clipByGlobalNorm, clipByGlobalNormObj, l2Loss } = require('./optimizers')

async function train (model, ds, config) {
    const defaultConfig = {
        epochs: 5,
        batchSize: 16,
        shuffle: true,
        lr: 6e-4,
        weightDecay: false,
        callbacks: []
    }
    config = Object.assign(defaultConfig, config || {})

    if (config.shuffle === true) {
        ds = ds.shuffle(config.batchSize * 10)
    } else if (config.shuffle === 'batch') {
        ds = ds.shuffle(config.batchSize)
    } else if (!isNaN(config.shuffle)) {
        ds = ds.shuffle(config.shuffle)
    }
    ds = ds.batch(config.batchSize)
    
    var includeInWeightDecay = []
    var excludeFromWeightDecay = []
    
    if (config.weightDecay === true) {
        config.weightDecay = 1e-4
    }
    if (config.weightDecay) {
        model.getNamedWeights().forEach(v => {
            if (
                v.name.includes('bias') 
                || v.name.includes('normalization') 
                || v.name.includes('emb') 
            ) {
                excludeFromWeightDecay.push(v.name)
            } else {
                includeInWeightDecay.push(v.name)
            }
        })
        var opt = new AdamW(config.lr, config.weightDecay, includeInWeightDecay, excludeFromWeightDecay)
    } else {
        var opt = tf.train.adam(6e-4)
    }
    
    for (let epoch = 1; epoch <= config.epochs; epoch++) {
        if (config.verbose) { console.log('\nEpoch', epoch) }
        
        var losses = []
        await ds.forEachAsync(({x, y}) => {
            var optFunc = () => {
                const logits = model.apply(x)
                const loss = tf.losses.softmaxCrossEntropy(y, logits)
                const lossValue = loss.arraySync()
                losses.push(lossValue)
                return loss
            }
            tf.tidy(() => {
                var {values, grads} = opt.computeGradients(optFunc)
                var gradsClipped = clipByGlobalNormObj(grads, 1)
                opt.applyGradients(gradsClipped)
            })
        })

        if (config.verbose) {
            console.log('Mem:', tf.memory().numTensors)
            console.log('Loss:', tf.tidy(() => tf.mean(losses).arraySync()))
        }
    }
}

module.exports = {
    train
}