const tf = require('@tensorflow/tfjs')
const { AdamW, clipByGlobalNorm, clipByGlobalNormObj, l2Loss } = require('./optimizers')

async function train (model, ds, config) {
    const defaultConfig = {
        epochs: null,
        maxIter: null,
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
        var opt = tf.train.adam(config.lr)
    }

    let epoch = 1
    let iteration = 1
    let iterator = await ds.iterator()

    while (true) {
        let next = await iterator.next()
        if (next.done) {
            epoch++
            if (config.epochs && epoch > config.epochs) {
                break
            }
            iterator = await ds.iterator()
            next = await iterator.next()
        }
        const {x, y} = next.value

        // Keep loss for reporting
        let loss
        const optFunc = () => {
            const logits = model.apply(x)
            loss = tf.keep(tf.losses.softmaxCrossEntropy(y, logits))
            return loss
        }
        tf.tidy(() => {
            let {values, grads} = opt.computeGradients(optFunc)
            let gradsClipped = clipByGlobalNormObj(grads, 1)
            opt.applyGradients(gradsClipped)
        })

        let lossVal = await loss.array()
        if (Array.isArray(config.callbacks)) {
            for (let callback of config.callbacks) {
                await callback(model, lossVal, iteration)
            }
        }

        // Dispose everything
        loss.dispose()
        x.dispose()
        y.dispose()

        // Check if we should stop
        iteration++
        if (config.maxIter && iteration > config.maxIter) {
            break
        }

        if (config.verbose) {
            console.log('Mem:', tf.memory())
            console.log(`Epoch: ${epoch}, Step: ${iteration}, Loss: ${lossVal}`)
        }

        await new Promise(resolve => setTimeout(resolve, 1))
    }

}

module.exports = {
    train
}