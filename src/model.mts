/* eslint-disable @typescript-eslint/no-unused-vars */
import * as tf from "@tensorflow/tfjs-node";
import { train } from "./train.mjs";

// Range Layer

interface RangeLayerConfig {
  name?: string;
}

class RangeLayer extends tf.layers.Layer {
  constructor(config: RangeLayerConfig) {
    super(config);
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    return inputShape;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);
      const [, T] = input.shape;
      const range = tf.reshape(tf.range(0, T, 1, "int32"), [1, T]); // .tile([B, 1])
      return range;
    });
  }

  static className: string = "RangeLayer";
}

// LogLayer

interface LogLayerConfig {
  name: string;
}

class LogLayer extends tf.layers.Layer {
  private config: LogLayerConfig;

  constructor(config: LogLayerConfig) {
    super(config);

    this.config = config;
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    return inputShape;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any = {}): tf.Tensor {
    return tf.tidy(() => {
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);
      const x = tf.util.flatten(input.arraySync());
      console.log(this.config.name + ">", input.shape, x[0], x[x.length - 1]);
      return input;
    });
  }

  static className: string = "LogLayer";
}

// CausalSelfAttentionBase Layer

interface CausalSelfAttentionBaseLayerConfig {
  blockSize: number;
  nEmbd: number;
  nHead: number;
  dropout: number;
  name?: string;
}

export class CausalSelfAttentionBaseLayer extends tf.layers.Layer {
  private config: CausalSelfAttentionBaseLayerConfig;
  private blockSize: number;
  private nEmbd: number;
  private nHead: number;
  private dropout: number;
  private mask: tf.Tensor;

  constructor(config: CausalSelfAttentionBaseLayerConfig) {
    super(config);
    this.config = config;
    this.blockSize = config.blockSize;
    this.nEmbd = config.nEmbd;
    this.nHead = config.nHead;
    this.dropout = config.dropout;
    this.mask = tf.linalg.bandPart(
      tf.ones([config.blockSize, config.blockSize]),
      -1,
      0,
    );
  }

  computeOutputShape(_inputShape: tf.Shape): tf.Shape {
    // Input here is already passed through a dense layer
    // It's shape is [B, T, 3 * nEmbd]
    // 3 there is for k, q, v (same as in MinGPT)
    // The output is [B, T, nEmbd]
    return [null, this.blockSize, this.nEmbd];
  }

  getConfig() {
    const config = super.getConfig();
    return Object.assign({}, config, this.config);
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      // Take into account that the input can be an array of tensors
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);

      // split() in TFJS requires a constant value for n splits
      // split() in Pytorch requires the size of each split
      let [q, k, v] = tf.split(input, 3, -1);
      const [B, T, C] = k.shape;
      const splitHeads = (x: tf.Tensor) =>
        tf.transpose(
          tf.reshape(x, [B, T, this.nHead, C / this.nHead]),
          [0, 2, 1, 3],
        );
      q = splitHeads(q);
      k = splitHeads(k);
      v = splitHeads(v);

      // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
      let att = tf.mul(
        tf.matMul(q, k, false, true),
        tf.div(1, tf.sqrt(tf.cast(k.shape[k.shape.length - 1], "float32"))),
      );
      att = tf.add(att, tf.mul(tf.sub(1, this.mask), -1e9));
      att = tf.softmax(att, -1);
      att = kwargs["training"] ? tf.dropout(att, this.dropout) : att;

      let y = tf.matMul(att, v);
      y = tf.transpose(y, [0, 2, 1, 3]);
      y = tf.reshape(y, [B, T, C]);

      return y;
    });
  }

  static className: string = "CausalSelfAttentionBaseLayer";
}

// CausalSelfAttentionLayer Layer

interface CausalSelfAttentionLayerConfig
  extends CausalSelfAttentionBaseLayerConfig {
  debug?: boolean;
  bias: boolean;
}

export class CausalSelfAttentionLayer extends tf.layers.Layer {
  private config: CausalSelfAttentionLayerConfig;
  private nEmbd: number;
  private nHead: number;
  private dropout: number;
  private bias: boolean;
  private mask: tf.Tensor;
  private cAttnKernel!: tf.LayerVariable;
  private cAttnBias!: tf.LayerVariable;
  private cProjKernel!: tf.LayerVariable;
  private cProjBias!: tf.LayerVariable;

  constructor(config: CausalSelfAttentionLayerConfig) {
    super(config);
    this.config = Object.assign({ name: "attn" }, config);

    // Config
    this.nEmbd = config.nEmbd;
    this.nHead = config.nHead;
    this.dropout = config.dropout;
    this.bias = config.bias;

    // Causal mask
    this.mask = tf.linalg.bandPart(
      tf.ones([config.blockSize, config.blockSize]),
      -1,
      0,
    );
  }

  build(_inputShape: tf.Shape): void {
    this.cAttnKernel = this.addWeight(
      "c_attn/kernel",
      [this.nEmbd, 3 * this.nEmbd],
      "float32",
      tf.initializers.glorotNormal({}),
    );
    this.cAttnBias = this.addWeight(
      "c_attn/bias",
      [3 * this.nEmbd],
      "float32",
      tf.initializers.zeros(),
    );
    this.cProjKernel = this.addWeight(
      "c_proj/kernel",
      [this.nEmbd, this.nEmbd],
      "float32",
      tf.initializers.glorotNormal({}),
    );
    this.cProjBias = this.addWeight(
      "c_proj/bias",
      [this.nEmbd],
      "float32",
      tf.initializers.zeros(),
    );
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    // console.log('computeOutputShape', inputShape)
    return inputShape;
    // return [null, this.blockSize, this.nEmbd]
  }

  getConfig() {
    // This is neeed to save and load the model
    // When the model is saved, the config is saved with it
    // When the model is loaded, the config is used to create a new instance of the layer
    const config = super.getConfig();
    return Object.assign({}, config, this.config);
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);

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

      const dense = (
        x: tf.Tensor,
        kernel: tf.LayerVariable,
        bias: tf.LayerVariable,
      ) => {
        const k = kernel.read().expandDims(0).tile([x.shape[0], 1, 1]);
        const m = tf.matMul(x, k);
        if (this.bias) {
          return tf.add(m, bias.read());
        } else {
          return m;
        }
      };

      const cAttn = dense(input, this.cAttnKernel, this.cAttnBias);

      // Make prder of qkv split to follow minGPT
      let [q, k, v] = tf.split(cAttn, 3, -1);
      const [B, T, C] = k.shape;

      if (this.config.debug) {
        new LogLayer({ name: "att_x" }).call(input);
        new LogLayer({ name: "att_c_attn" }).call(cAttn);
        new LogLayer({ name: "att_q_before" }).call(q);
        new LogLayer({ name: "att_k_before" }).call(k);
        new LogLayer({ name: "att_v_before" }).call(v);
      }

      const splitHeads = (x: tf.Tensor) =>
        tf.transpose(
          tf.reshape(x, [B, T, this.nHead, C / this.nHead]),
          [0, 2, 1, 3],
        );

      q = splitHeads(q);
      k = splitHeads(k);
      v = splitHeads(v);

      if (this.config.debug) {
        new LogLayer({ name: "att_q_after" }).call(q);
        new LogLayer({ name: "att_k_after" }).call(k);
        new LogLayer({ name: "att_v_after" }).call(v);
      }

      // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
      let att = tf.mul(
        tf.matMul(q, k, false, true),
        tf.div(1, tf.sqrt(tf.cast(k.shape[k.shape.length - 1], "float32"))),
      );

      const mask = this.mask.slice([0, 0], [T, T]);
      att = tf.add(att, tf.mul(tf.sub(1, mask), -1e9));
      att = tf.softmax(att, -1);
      att = kwargs["training"] ? tf.dropout(att, this.dropout) : att;
      if (this.config.debug) {
        new LogLayer({ name: "> att_softmax" }).call(att);
      }

      let y = tf.matMul(att, v);
      if (this.config.debug) {
        new LogLayer({ name: "att_yv" }).call(y);
      }

      y = tf.transpose(y, [0, 2, 1, 3]);
      y = tf.reshape(y, [B, T, C]);
      y = dense(y, this.cProjKernel, this.cProjBias);
      y = kwargs["training"] ? tf.dropout(y, this.dropout) : y;
      if (this.config.debug) {
        new LogLayer({ name: "att_y" }).call(y);
      }

      return y;
    });
  }

  static className: string = "CausalSelfAttentionLayer";
}

// GELU Activation Layer

export class GELULayer extends tf.layers.Layer {
  constructor() {
    super({});
  }

  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    return inputShape;
  }

  call(input: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor {
    return tf.tidy(() => {
      // In functional API, input is an array of tensors
      // So we need to get the first element (the actual input)
      // Add a check as here:
      // https://github.com/tensorflow/tfjs-examples/blob/master/custom-layer/custom_layer.js
      if (Array.isArray(input)) {
        input = input[0];
      }
      this.invokeCallHook(input, kwargs);
      const cdf = tf.mul(
        0.5,
        tf.add(
          1,
          tf.tanh(
            tf.mul(
              tf.sqrt(tf.div(2, Math.PI)),
              tf.add(input, tf.mul(0.044715, tf.pow(input, 3))),
            ),
          ),
        ),
      );
      return tf.mul(input, cdf);
    });
  }

  static className: string = "GELULayer";
}

// MLP Block

interface MLPConfig {
  blockSize: number;
  nEmbd: number;
  vocabSize: number;
  name?: string;
  residDrop: number;
}

export function MLP(conf: MLPConfig): tf.LayersModel {
  const config = Object.assign({ name: "mlp" }, conf);
  const inputs = tf.input({ shape: [config.blockSize, config.nEmbd] });
  let x;
  x = tf.layers
    .dense({
      name: config.name + "/c_fc",
      units: 4 * config.nEmbd,
      inputDim: config.nEmbd,
      inputShape: [config.blockSize, config.nEmbd],
    })
    .apply(inputs);
  x = new GELULayer().apply(x);
  x = tf.layers
    .dense({
      name: config.name + "/c_proj",
      units: config.nEmbd,
      inputDim: 4 * config.nEmbd,
      inputShape: [config.blockSize, 4 * config.nEmbd],
    })
    .apply(x);
  x = tf.layers
    .dropout({
      name: config.name + "/drop",
      rate: config.residDrop,
    })
    .apply(x) as tf.SymbolicTensor;
  return tf.model({ inputs: inputs, outputs: x });
}

// Block

interface BlockConfig extends MLPConfig, CausalSelfAttentionBaseLayerConfig {
  debug?: boolean;
  name?: string;
}

export function Block(conf: BlockConfig): tf.LayersModel {
  const config = Object.assign({ name: "h" }, conf);
  const inputs = tf.input({ shape: [config.blockSize, config.nEmbd] });
  let x1: tf.SymbolicTensor, x2: tf.SymbolicTensor;
  // Attention
  // Setting epsilon to 1e-5 for LayerNorms to be consistent with PyTorch
  x1 = tf.layers
    .layerNormalization({ name: config.name + "/ln_1", epsilon: 1e-5 })
    .apply(inputs) as tf.SymbolicTensor;
  if (config.debug) {
    x1 = new LogLayer({ name: config.name + "/ln_1_log" }).apply(
      x1,
    ) as tf.SymbolicTensor;
  }
  x1 = new CausalSelfAttentionLayer({
    ...config,
    name: config.name + "/attn",
    bias: true,
  }).apply(x1) as tf.SymbolicTensor;
  x1 = tf.layers.add().apply([inputs, x1]) as tf.SymbolicTensor;
  // MLP
  x2 = tf.layers
    .layerNormalization({ name: config.name + "/ln_2", epsilon: 1e-5 })
    .apply(x1) as tf.SymbolicTensor;
  x2 = MLP(Object.assign({}, config, { name: config.name + "/mlp" })).apply(
    x2,
  ) as tf.SymbolicTensor;
  x2 = tf.layers.add().apply([x1, x2]) as tf.SymbolicTensor;
  return tf.model({
    name: config.name,
    inputs: inputs,
    outputs: x2,
  }) as tf.LayersModel;
}

// GPT Model

interface GPTConfig extends BlockConfig {
  name?: string;
  bias?: boolean;
  debug?: boolean;
  tokEmb?: boolean;
  lmHead?: boolean;
  embdDrop?: number;
  nLayer?: number;
  modelType:
    | "gpt2"
    | "gpt2-medium"
    | "gpt2-large"
    | "gpt2-xl"
    | "gpt-mini"
    | "gpt-micro"
    | "gpt-nano";
}

export function GPT(conf: Partial<GPTConfig>): tf.LayersModel {
  const configDefaults = {
    name: "transformer",
    bias: true,
    debug: false,
    tokEmb: true,
    lmHead: true,
  };
  const configModels = {
    gpt2: {
      nLayer: 12,
      nHead: 12,
      nEmbd: 768,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt2-medium": {
      nLayer: 24,
      nHead: 16,
      nEmbd: 1024,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt2-large": {
      nLayer: 36,
      nHead: 20,
      nEmbd: 1280,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt2-xl": {
      nLayer: 48,
      nHead: 25,
      nEmbd: 1600,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt-mini": {
      nLayer: 6,
      nHead: 6,
      nEmbd: 192,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt-micro": {
      nLayer: 4,
      nHead: 4,
      nEmbd: 128,
      vocabSize: 50257,
      blockSize: 1024,
    },
    "gpt-nano": {
      nLayer: 3,
      nHead: 3,
      nEmbd: 48,
      vocabSize: 50257,
      blockSize: 1024,
    },
  };
  // Check if modelType is present in conf
  if (conf.modelType) {
    // If so, check if it's valid
    if (!Object.keys(configModels).includes(conf.modelType)) {
      throw new Error(`Invalid modelType: ${conf.modelType}`);
    }
    // If valid, merge modelConfig with configDefaults
    const modelConfig = configModels[conf.modelType];
    Object.assign(configDefaults, modelConfig);
  }

  const config = Object.assign({}, configDefaults, conf) as Required<GPTConfig>;

  const inputs = tf.input({ shape: [null] });

  const tokEmb = config.tokEmb
    ? tf.layers
        .embedding({
          name: config.name + "/wte",
          inputDim: config.vocabSize as number,
          outputDim: config.nEmbd as number,
          embeddingsInitializer: "zeros",
        })
        .apply(inputs)
    : inputs;

  const range = new RangeLayer({}).apply(inputs);
  let posEmb = tf.layers
    .embedding({
      name: config.name + "/wpe",
      inputDim: config.blockSize as number,
      outputDim: config.nEmbd as number,
      embeddingsInitializer: "zeros",
    })
    .apply(range);
  if (config.debug) {
    posEmb = new LogLayer({ name: "posEmb" }).apply(posEmb);
  }

  let x;
  x = tf.layers.add().apply([tokEmb, posEmb] as tf.SymbolicTensor[]);
  x = tf.layers
    .dropout({
      name: "drop",
      rate: config.embdDrop as number,
    })
    .apply(x);
  if (config.debug) {
    x = new LogLayer({ name: "dropadd" }).apply(x);
  }

  for (let i = 0; i < (config.nLayer as number); i++) {
    x = Block(
      Object.assign({}, config, { name: config.name + "/h/" + i }),
    ).apply(x);
  }
  x = tf.layers
    .layerNormalization({ name: config.name + "/ln_f", epsilon: 1e-5 })
    .apply(x);
  if (config.debug) {
    x = new LogLayer({ name: "fin/ln" }).apply(x);
  }

  if (config.lmHead) {
    x = tf.layers
      .dense({
        name: "lm_head",
        units: config.vocabSize,
        inputDim: config.nEmbd,
        inputShape: [config.blockSize, config.nEmbd],
        useBias: false,
      })
      .apply(x);
  }
  return tf.model({ inputs: inputs, outputs: x } as {
    inputs: tf.SymbolicTensor;
    outputs: tf.SymbolicTensor;
  });
}

// GPTModel

const defaultGenerateConfig: GPTLMHeadModelGenerateConfig = {
  maxNewTokens: 20,
  temperature: 1.0,
  doSample: false,
  topK: 1,
};

function prepareIdx(idx: number[] | tf.Tensor): tf.Tensor {
  tf.tidy(() => {
    // Check if idx is a tensor or an array
    if (idx instanceof tf.Tensor) {
      idx = idx.clone();
    } else {
      idx = tf.tensor(idx);
    }
    // Check data type
    if (idx.dtype !== "int32") {
      idx = idx.toInt();
    }
    // If the shape of idx is 1D, we need to add a dimension
    if (idx.shape.length === 1) {
      idx = idx.expandDims(0);
    }
    tf.keep(idx);
    // keep idx from deletion
  });
  return idx as tf.Tensor;
}

function generateOnce(
  model: tf.LayersModel,
  idx: tf.Tensor,
  config: GPTLMHeadModelGenerateConfig,
) {
  let idxNext: tf.Tensor<tf.Rank> | null = null;
  let timePerToken = performance.now();
  tf.tidy(() => {
    const block_size = model.inputs[0].shape[1] as number;

    const idxCond =
      (idx.shape[1] as number) <= block_size
        ? idx
        : idx.slice([0, -block_size], [-1, -1]);
    // Forward the model to get the logits for the index in the sequence
    const logits = model.predict(idxCond) as tf.Tensor;
    timePerToken = performance.now() - timePerToken;
    // pluck the logits at the final step and scale by desired temperature
    const logitsScaled = logits
      .slice([0, (idx.shape[1] as number) - 1, 0])
      .reshape([logits.shape[0], logits.shape[2] as number])
      .div(tf.scalar(config.temperature));
    // TODO: topK sampling
    // apply softmax to convert logits to (normalized) probabilities
    const probs = logitsScaled.softmax(-1) as tf.Tensor<tf.Rank.R2>;
    // either sample from the distribution or take the most likely element
    if (config.doSample) {
      idxNext = tf.multinomial(probs, 1);
    } else {
      idxNext = probs.argMax(-1);
      idxNext = idxNext.expandDims(1);
    }
    tf.keep(idxNext);
  });
  return {
    idxNext: idxNext as never as tf.Tensor<tf.Rank>,
    timePerToken,
  };
}

export function generateSync(
  model: tf.LayersModel,
  idx: tf.Tensor,
  conf: GPTLMHeadModelGenerateConfig,
  callback?: (data: GPTLMHeadModelGenerateCallbackData) => void,
) {
  const config = Object.assign({}, defaultGenerateConfig, conf);
  idx = prepareIdx(idx);
  for (let step = 0; step < config.maxNewTokens; step++) {
    const { idxNext, timePerToken } = generateOnce(model, idx, config);
    const idxNew = idx.concat(idxNext, 1);
    tf.dispose(idx);
    idx = idxNew;
    const idxNextArr = idxNext.arraySync() as number[];
    tf.dispose(idxNext);
    if (callback) {
      callback({ idxNext: idxNextArr, timePerToken: timePerToken });
    }
  }
  const idxArr = idx.arraySync();
  tf.dispose(idx);
  return idxArr as number[];
}

export async function generate(
  model: tf.LayersModel,
  idx: number[] | tf.Tensor,
  conf: GPTLMHeadModelGenerateConfig,
  callback?: (data: GPTLMHeadModelGenerateCallbackData) => void,
) {
  const config = Object.assign({}, defaultGenerateConfig, conf);
  idx = prepareIdx(idx);
  for (let step = 0; step < config.maxNewTokens; step++) {
    const { idxNext, timePerToken } = generateOnce(model, idx, config);
    const idxNew = idx.concat(idxNext, 1) as tf.Tensor;
    tf.dispose(idx);
    idx = idxNew;
    const idxNextArr = (await idxNext.array()) as number[];
    tf.dispose(idxNext);
    if (callback) {
      callback({ idxNext: idxNextArr, timePerToken: timePerToken });
    }
  }
  const idxArr = await idx.array();
  tf.dispose(idx);
  return idxArr as number[][];
}

class GPTModel_ {
  protected config: GPTConfig;
  protected model: tf.LayersModel;

  constructor(config: GPTConfig) {
    this.config = config;
    this.model = GPT(config);
  }

  async load(weights: tf.NamedTensorMap) {
    await this.model.loadWeights(weights);
  }

  async save(modelPath: string) {
    await this.model.save(modelPath);
  }

  apply(inputs: tf.Tensor | tf.Tensor[]) {
    return this.model.apply(inputs);
  }

  predict(inputs: tf.Tensor | tf.Tensor[]) {
    return this.model.predict(inputs);
  }
}
// GPTLMHeadModel

interface GPTLMHeadModelGenerateConfig {
  maxNewTokens: number;
  temperature: number;
  topK: number;
  doSample: boolean;
}

interface GPTLMHeadModelGenerateCallbackData {
  idxNext: number[];
  timePerToken: number;
}

type GPTLMHeadModelGenerateCallback = (
  data: GPTLMHeadModelGenerateCallbackData,
) => void | Promise<void>;

export class GPTLMHeadModel extends GPTModel_ {
  constructor(config: GPTConfig) {
    super(config);
  }

  async train(dataset: any, config: any): Promise<void> {
    await train(this.model, dataset, config);
  }

  async generate(
    config: GPTLMHeadModelGenerateConfig,
    callback?: GPTLMHeadModelGenerateCallback,
  ): Promise<number[][]> {
    return await generate(this.model, null, config, callback);
  }

  generateSync(
    config: GPTLMHeadModelGenerateConfig,
    callback?: GPTLMHeadModelGenerateCallback,
  ): number[][] {
    return generateSync(this.model, null, config, callback);
  }
}
