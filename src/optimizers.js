
const tf = require('@tensorflow/tfjs')

var ENGINE = tf.engine()

function l2Loss(tensor) {
    // https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    return tf.div(tf.sum(tf.square(tensor)), 2)
}

function globalNorm(tensors) {
    // https://github.com/tensorflow/tensorflow/blob/c256c071bb26e1e13b4666d1b3e229e110bc914a/tensorflow/python/ops/clip_ops.py#L242
    var halfSquaredNorms = []
    tensors.forEach((tensor, ti) => {
        halfSquaredNorms.push(l2Loss(tensor))
    })
    var halfSquaredNorm = tf.sum(tf.stack(halfSquaredNorms))
    var norm = tf.sqrt(tf.mul(halfSquaredNorm, tf.scalar(2.0, halfSquaredNorm.dtype)))
    return norm
}

function clipByGlobalNorm(tensors, clipNorm, useNorm) {
    // https://github.com/kamalkraj/minGPT-TF/blob/master/mingpt/optimization.py
    // https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/clip_ops.py#L291-L382
    /*
    To perform the clipping, the values t_list[i] are set to:
        t_list[i] * clip_norm / max(global_norm, clip_norm)
    where:
        global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
    */
    var useNorm = useNorm || globalNorm(tensors)
    var scale = tf.mul(
        clipNorm,
        tf.minimum(
            tf.div(tf.scalar(1.0), useNorm),
            tf.div(tf.scalar(1.0, dtype=useNorm.dtype), clipNorm)
        )
    )
    var tensorsClipped = []
    tensors.forEach((tensor, ti) => {
        tensorsClipped.push(
            tf.clone(tf.mul(tensor, scale))
        )
    })
    return tensorsClipped
}

function clipByGlobalNormObj(tensorsObj, clipNorm, useNorm) {
    const varNames = Object.keys(tensorsObj)
    const tensorsArr = varNames.map(n => tensorsObj[n])
    const tensorsArrClipped = clipByGlobalNorm(tensorsArr, clipNorm, useNorm)
    const tensorsObjClipped = {}
    tensorsArrClipped.forEach((t, ti) => {
        tensorsObjClipped[varNames[ti]] = t
    })
    return tensorsObjClipped
}

var AdamW = class extends tf.AdamOptimizer {
    constructor(params) {
        var defaultParams = {
            learningRate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-07,
            weightDecayRate: 0,
            includeInWeightDecay: [],
            excludeFromWeightDecay: [],
            gradientClipNorm: 1.0
        }
        var p = Object.assign({}, defaultParams, params)
        super(p.learningRate, p.beta1, p.beta2, p.epsilon)
        // this.learningRate = p.learningRate
        this.weightDecayRate = p.weightDecayRate
        this.includeInWeightDecay = p.includeInWeightDecay
        this.excludeFromWeightDecay = p.excludeFromWeightDecay
        this.gradientClipNorm = p.gradientClipNorm
    }
    applyGradients(variableGradients) {
        // log(variableGradients, typeof variableGradients)
        const varNames = Array.isArray(variableGradients)
            ? variableGradients.map(v => v.name)
            : Object.keys(variableGradients)
        // log(this.learningRate, varNames)

        // const varValues = varNames.map(n => variableGradients[n])
        // const varValuesClipped = clipByGlobalNorm(varValues, 1)
        // varValuesClipped.forEach((v, i) => variableGradients[varNames[i]] = v)

        // Apply weight decay
        varNames.forEach((name, i) => {
            if (this.includeInWeightDecay.includes(name)) {
                const value = ENGINE.registeredVariables[name]
                const newValue = tf.sub(value, tf.mul(this.learningRate, tf.mul(value, this.weightDecayRate)))
                value.assign(newValue)
            }
        })

        super.applyGradients(variableGradients)

        /*
        varNames.forEach((name, i) => {
            // variableGradients[name] = variableGradients[name].clipByValue(-1, 1)
            // log(name, variableGradients[name].arraySync(), typeof variableGradients[name])
            if (this._include_in_weight_decay.includes(name)) {
                var grads = {}
                grads[name] = variableGradients[name]
                super.applyGradients(grads)
            }
        })
        */
    }
}

module.exports = {
    AdamW,
    clipByGlobalNorm,
    clipByGlobalNormObj
}


    