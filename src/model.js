const tf = require('@tensorflow/tfjs')
const { train } = require('./train')

const Range = config => new Range_(config)
class Range_ extends tf.layers.Layer {
    computeOutputShape(inputShape) { 
        return inputShape
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            if (Array.isArray(input)) {
                input = input[0]
            }
            this.invokeCallHook(input, kwargs)
            const [B, T] = input.shape
            const range = tf.reshape(tf.range(0, T, 1, 'int32'), [1, T]) // .tile([B, 1])
            return range
        })
    }

    static get className() {
        return 'Range'
    }
}
tf.serialization.registerClass(Range_)

const LogLayer = config => new LogLayer_(config)
class LogLayer_ extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.config = config
    }

    computeOutputShape(inputShape) { 
        return inputShape
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            if (Array.isArray(input)) {
                input = input[0]
            }
            this.invokeCallHook(input, kwargs)
            const x = tf.util.flatten(input.arraySync())
            console.log(this.config.name + '>', input.shape, x[0], x[x.length-1])
            return input
        })
    }

    static get className() {
        return 'LogLayer'
    }
}
tf.serialization.registerClass(LogLayer_)

const CausalSelfAttentionBase = config => new CausalSelfAttentionBase_(config)
class CausalSelfAttentionBase_ extends tf.layers.Layer {
    constructor(config, i) {
        super(config)
        this.config = config
        this.blockSize = config.blockSize
        this.nEmbd = config.nEmbd
        this.nHead = config.nHead
        this.dropout = config.dropout
        this.i = i
        this.mask = tf.linalg.bandPart(tf.ones([config.blockSize, config.blockSize]), -1, 0)
    }

    computeOutputShape(inputShape) {
        // Input here is already passed through a dense layer
        // It's shape is [B, T, 3 * nEmbd]
        // 3 there is for k, q, v (same as in MinGPT)
        // The output is [B, T, nEmbd]
        return [null, this.blockSize, this.nEmbd]; 
    }

    getConfig() {
        const config = super.getConfig()
        return Object.assign({}, config, this.config)
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            // Take into account that the input can be an array of tensors
            if (Array.isArray(input)) {
                input = input[0]
            }
            this.invokeCallHook(input, kwargs)

            // split() in TFJS requires a constant value for n splits
            // split() in Pytorch requires the size of each split
            let [q, k, v] = tf.split(input, 3, -1)
            const [B, T, C] = k.shape
            const splitHeads = (x) => tf.transpose(
                tf.reshape(x, [B, T, this.nHead, C / this.nHead]),
                [0, 2, 1, 3]
            )
            q = splitHeads(q)
            k = splitHeads(k)
            v = splitHeads(v)
            
            // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            let att = tf.mul(
                tf.matMul(q, k, false, true), 
                tf.div(1, tf.sqrt(tf.cast(k.shape[k.shape.length - 1], 'float32')))
            )
            att = tf.add(att, tf.mul(tf.sub(1, this.mask), -1e9))
            att = tf.softmax(att, -1)
            att = kwargs['training'] ? tf.dropout(att, this.dropout) : att

            let y = tf.matMul(att, v)
            y = tf.transpose(y, [0, 2, 1, 3])
            y = tf.reshape(y, [B, T, C])

            return y
        })
    }

    static get className() {
        return 'CausalSelfAttentionBase'
    }
}
tf.serialization.registerClass(CausalSelfAttentionBase_)

function CausalSelfAttentionMixed(conf) {
    const config = Object.assign({ name: 'attn' }, conf)
    const csa = CausalSelfAttentionBase(config)
    const inputs = tf.input({shape: [config.blockSize, config.nEmbd]}) 
    let att
    att = tf.layers.dense({
            name: config.name + '/c_attn', 
            units: 3 * config.nEmbd, 
            inputDim: config.nEmbd, 
            inputShape: [config.blockSize, config.nEmbd], 
            useBias: config.bias
        })
        .apply(inputs)
    att = csa
        .apply(att)
    att = tf.layers.dense({
            name: config.name + '/proj', 
            units: config.nEmbd, 
            inputDim: config.nEmbd, 
            inputShape: [config.blockSize, config.nEmbd], 
            useBias: config.bias
        })
        .apply(att)
    att = tf.layers.dropout({
            name: config.name + '/drop', 
            rate: config.dropout
        })
        .apply(att)
    return tf.model({ inputs: inputs, outputs: att })
}

const CausalSelfAttention = config => new CausalSelfAttention_(config)
class CausalSelfAttention_ extends tf.layers.Layer {
    constructor(config) {
        super(config)
        this.config = Object.assign({ name: 'attn' }, config)

        // Config
        this.blockSize = config.blockSize
        this.nEmbd = config.nEmbd
        this.nHead = config.nHead
        this.dropout = config.dropout
        this.bias = config.bias 
        
        // Causal mask
        this.mask = tf.linalg.bandPart(tf.ones([config.blockSize, config.blockSize]), -1, 0)
    }
    
    build(inputShape) {
        this.cAttnKernel = this.addWeight('c_attn/kernel', [this.nEmbd, 3 * this.nEmbd], 'float32', tf.initializers.glorotNormal())
        this.cAttnBias = this.addWeight('c_attn/bias', [3 * this.nEmbd], 'float32', tf.initializers.zeros())
        this.cProjKernel = this.addWeight('c_proj/kernel', [this.nEmbd, this.nEmbd], 'float32', tf.initializers.glorotNormal())
        this.cProjBias = this.addWeight('c_proj/bias', [this.nEmbd], 'float32', tf.initializers.zeros())
    }

    computeOutputShape(inputShape) {
        // console.log('computeOutputShape', inputShape)
        return inputShape
        // return [null, this.blockSize, this.nEmbd]
    }

    getConfig() {
        // This is neeed to save and load the model
        // When the model is saved, the config is saved with it
        // When the model is loaded, the config is used to create a new instance of the layer
        const config = super.getConfig()
        return Object.assign({}, config, this.config)
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            if (Array.isArray(input)) {
                input = input[0]
            }
            this.invokeCallHook(input, kwargs)

            // Direct application of matMul to x and kernel throws:
            // > Error in gradient for op BatchMatMul. 
            // > The gradient of input 'b' has shape '16,48,48', 
            // > which does not match the shape of the input '48,48'
            // Two solutions worked:
            // 1. Use tf.layers.dense but reassign kernel and bias
            // 2. Use tf.matMul but expandDims and tile kernel (current)
            // Another option, of course, is to separate attention logic
            // from trainable weights completely and use tf.layers.dense
            // inside a model definition. I was not able to define fully
            // function regular dense layers inside a custom layer.
            // Something related to how weights are loaded with this.kernel
            // and duplicating names
            
            const dense = (x, kernel, bias) => {
                const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1])
                const m = tf.matMul(x, k)
                if (this.bias) {
                    return tf.add(m, bias.read())
                } else {
                    return m
                }
            }

            const cAttn = dense(input, this.cAttnKernel, this.cAttnBias)
            
            // Make prder of qkv split to follow minGPT
            let [q, k, v] = tf.split(cAttn, 3, -1)
            const [B, T, C] = k.shape

            if (this.config.debug) {
                LogLayer({ name: 'att_x' }).call(input)
                LogLayer({ name: 'att_c_attn' }).call(cAttn)
                LogLayer({ name: 'att_q_before' }).call(q)
                LogLayer({ name: 'att_k_before' }).call(k)
                LogLayer({ name: 'att_v_before' }).call(v)
            }

            const splitHeads = (x) => tf.transpose(
                tf.reshape(x, [B, T, this.nHead, C / this.nHead]),
                [0, 2, 1, 3]
            )

            q = splitHeads(q)
            k = splitHeads(k)
            v = splitHeads(v)

            if (this.config.debug) {
                LogLayer({ name: 'att_q_after' }).call(q)
                LogLayer({ name: 'att_k_after' }).call(k)
                LogLayer({ name: 'att_v_after' }).call(v)
            }
            
            // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            let att = tf.mul(
                tf.matMul(q, k, false, true), 
                tf.div(1, tf.sqrt(tf.cast(k.shape[k.shape.length - 1], 'float32')))
            )

            const mask = this.mask.slice([0, 0], [T, T])
            att = tf.add(att, tf.mul(tf.sub(1, mask), -1e9))
            att = tf.softmax(att, -1)
            att = kwargs['training'] ? tf.dropout(att, this.dropout) : att
            if (this.config.debug) { LogLayer({ name: '> att_softmax' }).call(att) }

            let y = tf.matMul(att, v)
            if (this.config.debug) { LogLayer({ name: 'att_yv' }).call(y) }

            y = tf.transpose(y, [0, 2, 1, 3])
            y = tf.reshape(y, [B, T, C])
            y = dense(y, this.cProjKernel, this.cProjBias)
            y = kwargs['training'] ? tf.dropout(y, this.dropout) : y
            if (this.config.debug) { LogLayer({ name: 'att_y' }).call(y) }

            return y
        })
    }

    static get className() {
        return 'CausalSelfAttention'
    }
}
tf.serialization.registerClass(CausalSelfAttention_);

const GELU = () => new GELU_()
class GELU_ extends tf.layers.Layer {
    constructor() {
        super({})
    }

    computeOutputShape(inputShape) {
        return inputShape  
    }

    call(input, kwargs) {
        return tf.tidy(() => {
            // In functional API, input is an array of tensors
            // So we need to get the first element (the actual input)
            // Add a check as here:
            // https://github.com/tensorflow/tfjs-examples/blob/master/custom-layer/custom_layer.js
            if (Array.isArray(input)) {
                input = input[0]
            }
            this.invokeCallHook(input, kwargs)
            const cdf = tf.mul( 
                0.5,
                tf.add(
                    1,
                    tf.tanh(tf.mul(
                        tf.sqrt(tf.div(2, Math.PI)),
                        tf.add(input, tf.mul(0.044715, tf.pow(input, 3)))
                    ))
                )
            )
            return tf.mul(input, cdf)
        })
    }

    static get className() {
        return 'GELU'
    }
}
tf.serialization.registerClass(GELU_)

function MLP(conf) {
    const config = Object.assign({ name: 'mlp' }, conf)
    const inputs = tf.input({shape: [config.blockSize, config.nEmbd]})
    let x 
    x = tf.layers.dense({
            name: config.name + '/c_fc', 
            units: 4 * config.nEmbd, 
            inputDim: config.nEmbd, 
            inputShape: [config.blockSize, config.nEmbd]
        })
        .apply(inputs)
    x = GELU()
        .apply(x)
    x = tf.layers.dense({
            name: config.name + '/c_proj', 
            units: config.nEmbd, 
            inputDim: 4 * config.nEmbd, 
            inputShape: [config.blockSize, 4 * config.nEmbd] 
        })
        .apply(x)
    x = tf.layers.dropout({
            name: config.name + '/drop', rate: config.residDrop
        })
        .apply(x)
    return tf.model({ inputs: inputs, outputs: x })
}

function Block(conf) {
    const config = Object.assign({ name: 'h' }, conf)
    const inputs = tf.input({shape: [config.blockSize, config.nEmbd]}) 
    let x1, x2 
    // Attention
    // Setting epsilon to 1e-5 for LayerNorms to be consistent with PyTorch
    x1 = tf.layers.layerNormalization({name: config.name + '/ln_1', epsilon: 1e-5})
        .apply(inputs)
    if (config.debug) { x1 = LogLayer({name: config.name + '/ln_1_log'}).apply(x1) }
    x1 = CausalSelfAttention(Object.assign({}, config, {name: config.name + '/attn'}))
        .apply(x1)
    x1 = tf.layers.add()
        .apply([inputs, x1])
    // MLP
    x2 = tf.layers.layerNormalization({name: config.name + '/ln_2', epsilon: 1e-5})
        .apply(x1)
    x2 = MLP(Object.assign({}, config, {name: config.name + '/mlp'}))
        .apply(x2)
    x2 = tf.layers.add()
        .apply([x1, x2])
    return tf.model({name: config.name, inputs: inputs, outputs: x2})
}

function GPT(conf) {
    const configDefaults = {
        name: 'transformer',
        bias: true,
        debug: false,
        tokEmb: true,
        lmHead: true,
    }
    const configModels = {
        'gpt2': { nLayer: 12, nHead: 12, nEmbd: 768, vocabSize: 50257, blockSize: 1024 },
        'gpt2-medium': { nLayer: 24, nHead: 16, nEmbd: 1024, vocabSize: 50257, blockSize: 1024 },
        'gpt2-large': { nLayer: 36, nHead: 20, nEmbd: 1280, vocabSize: 50257, blockSize: 1024 },
        'gpt2-xl': { nLayer: 48, nHead: 25, nEmbd: 1600, vocabSize: 50257, blockSize: 1024 },
        'gpt-mini': { nLayer: 6, nHead: 6, nEmbd: 192 },
        'gpt-micro': { nLayer: 4, nHead: 4, nEmbd: 128 },
        'gpt-nano': { nLayer: 3, nHead: 3, nEmbd: 48 },
    }
    // Check if modelType is present in conf
    if (conf.modelType) {
        // If so, check if it's valid
        if (!Object.keys(configModels).includes(conf.modelType)) {
            throw new Error(`Invalid modelType: ${conf.modelType}`)
        }
        // If valid, merge modelConfig with configDefaults
        const modelConfig = configModels[conf.modelType]
        Object.assign(configDefaults, modelConfig)
    }

    const config = Object.assign({}, configDefaults, conf)

    const inputs = tf.input({shape: [null]})
    
    const tokEmb = config.tokEmb 
        ? tf.layers.embedding({
                name: config.name + '/wte', 
                inputDim: config.vocabSize, 
                outputDim: config.nEmbd,
                embeddingsInitializer: 'zeros',
                embeddingsRegularizer: null,
                activityRegularizer: null,
            })
            .apply(inputs)
        : inputs

    const range = Range()
        .apply(inputs)
    const posEmb = tf.layers.embedding({
            name: config.name + '/wpe',
            inputDim: config.blockSize,
            outputDim: config.nEmbd,
            embeddingsInitializer: 'zeros'
        })
        .apply(range)
    if (config.debug) { posEmb = LogLayer({name: 'posEmb'}).apply(posEmb) }

    let x 
    x = tf.layers.add()
        .apply([tokEmb, posEmb])
    x = tf.layers.dropout({
            name: 'drop', 
            rate: config.embdDrop
        })
        .apply(x)
    if (config.debug) { x = LogLayer({name: 'dropadd'}).apply(x) }

    for (let i = 0; i < config.nLayer; i++) {
        x = Block(Object.assign({}, config, {name: config.name + '/h/' + i})).apply(x)
    }
    x = tf.layers.layerNormalization({name: config.name + '/ln_f', epsilon: 1e-5})
        .apply(x)
    if (config.debug) { x = LogLayer({name: 'fin/ln'}).apply(x) }

    if (config.lmHead) {
        x = tf.layers.dense({
                name: 'lm_head', 
                units: config.vocabSize, 
                inputDim: config.nEmbd, 
                inputShape: [config.blockSize, config.nEmbd],
                useBias: false
            })
            .apply(x)
    }
    return tf.model({inputs: inputs, outputs: x})
}

const defaultGenerateConfig = {
    maxNewTokens: 20,
    temperature: 1.0,
    doSample: false,
    topK: null,
}

function prepareIdx(idx) {
    tf.tidy(() => {
        // Check if idx is a tensor or an array
        if (idx instanceof tf.Tensor) {
            idx = idx.clone()
        } else {
            idx = tf.tensor(idx)
        }
        // Check data type
        if (idx.dtype !== 'int32') {
            idx = idx.toInt()
        }
        // If the shape of idx is 1D, we need to add a dimension
        if (idx.shape.length === 1) {
            idx = idx.expandDims(0)
        }
        tf.keep(idx)
        // keep idx from deletion
    })
    return idx
}

function generateOnce(model, idx, config) {
    let idxNext
    let timePerToken = performance.now()
    tf.tidy(() => {
        const block_size = model.inputs[0].shape[1]
        const idxCond = idx.shape[1] <= block_size ? idx : idx.slice([0, -block_size], [-1, -1])
        // Forward the model to get the logits for the index in the sequence
        const logits = model.predict(idxCond)
        timePerToken = performance.now() - timePerToken
        // pluck the logits at the final step and scale by desired temperature
        const logitsScaled = logits
            .slice([0, idx.shape[1] - 1, 0])
            .reshape([logits.shape[0], logits.shape[2]])
            .div(tf.scalar(config.temperature))
        // TODO: topK sampling
        // apply softmax to convert logits to (normalized) probabilities
        const probs = logitsScaled.softmax(-1)
        // either sample from the distribution or take the most likely element
        if (config.doSample) {
            idxNext = tf.multinomial(probs, 1)
        } else {
            idxNext = probs.argMax(-1)
            idxNext = idxNext.expandDims(1)
        }
        tf.keep(idxNext)
    })
    return {
        idxNext,
        timePerToken
    }
}

function generateSync(model, idx, conf, callback) {
    const config = Object.assign({}, defaultGenerateConfig, conf)
    idx = prepareIdx(idx)
    for (let step = 0; step < config.maxNewTokens; step++) {
        const { idxNext, timePerToken } = generateOnce(model, idx, config)
        const idxNew = idx.concat(idxNext, 1)
        tf.dispose(idx)
        idx = idxNew
        const idxNextArr = idxNext.arraySync()
        tf.dispose(idxNext)
        if (callback) {
            callback({ idxNext: idxNextArr, timePerToken: timePerToken });
        }
    }
    const idxArr = idx.arraySync()
    tf.dispose(idx)
    return idxArr
}

async function generate(model, idx, conf, callback) {
    const config = Object.assign({}, defaultGenerateConfig, conf)
    idx = await prepareIdx(idx)
    for (let step = 0; step < config.maxNewTokens; step++) {
        const { idxNext, timePerToken } = generateOnce(model, idx, config)
        const idxNew = idx.concat(idxNext, 1)
        tf.dispose(idx)
        idx = idxNew
        const idxNextArr = await idxNext.array()
        tf.dispose(idxNext)
        if (callback) {
            await callback({ idxNext: idxNextArr, timePerToken: timePerToken })
        }
    }
    const idxArr = await idx.array()
    tf.dispose(idx)
    return idxArr
}

const GPTModel = config => new GPTModel_(config)
class GPTModel_ {
    constructor(config) {
        this.config = config
        this.model = GPT(config)
    }

    async load(modelPath) {
        await this.model.loadWeights(modelPath)
    }

    async save(modelPath) {
        await this.model.save(modelPath)
    }
    
    apply(inputs) {
        return this.model.apply(inputs)
    }

    predict(inputs) {
        return this.model.predict(inputs)
    }
}

const GPTLMHeadModel = config => new GPTLMHeadModel_(config)
class GPTLMHeadModel_ extends GPTModel_ {
    constructor(config) {
        super(config)
    }

    async train(dataset, config) {
        await train(this.model, dataset, config)
    }

    async generate() {
        return await generate(this.model, ...arguments)
    }

    generateSync() {
        return generateSync(this.model, ...arguments)
    }
}

module.exports = {
    GELU,
    CausalSelfAttention,
    CausalSelfAttentionMixed,
    MLP,
    Block,
    GPT,
    GPTModel,
    GPTLMHeadModel,
    generate, generateSync
}
