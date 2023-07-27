import * as tf from "@tensorflow/tfjs-node";
import { AdamW, clipByGlobalNormObj } from "./optimizers.mjs";

export async function train(
  model: tf.LayersModel,
  ds: tf.data.Dataset<tf.TensorContainer>,
  config: {
    epochs?: number;
    maxIter?: number;
    batchSize?: number;
    shuffle?: boolean | number | "batch";
    lr?: number;
    weightDecay?: boolean | number;
    callbacks?: any[];
    verbose?: boolean;
  } = {},
) {
  if (undefined === config.batchSize) {
    config.batchSize = 16;
  }

  if (undefined === config.lr) {
    config.lr = 6e-4;
  }

  if (undefined === config.shuffle) {
    config.shuffle = true;
  }

  if (undefined === config.weightDecay) {
    config.weightDecay = false;
  }

  if (undefined === config.callbacks) {
    config.callbacks = [];
  }

  if (config.shuffle === true) {
    ds = ds.shuffle(config.batchSize * 10);
  } else if (config.shuffle === "batch") {
    ds = ds.shuffle(config.batchSize);
  } else if (false !== config.shuffle && !isNaN(config.shuffle)) {
    ds = ds.shuffle(config.shuffle);
  }
  ds = ds.batch(config.batchSize);

  const includeInWeightDecay: string[] = [];
  const excludeFromWeightDecay: string[] = [];

  if (config.weightDecay === true) {
    config.weightDecay = 1e-4;
  }
  let opt = tf.train.adam(config.lr);
  if (config.weightDecay) {
    model["getNamedWeights"]().forEach((v) => {
      if (
        v.name.includes("bias") ||
        v.name.includes("normalization") ||
        v.name.includes("emb")
      ) {
        excludeFromWeightDecay.push(v.name);
      } else {
        includeInWeightDecay.push(v.name);
      }
    });
    opt = new AdamW({
      learningRate: config.lr,
      weightDecayRate: config.weightDecay,
      includeInWeightDecay,
      excludeFromWeightDecay,
    });
  }

  let epoch = 1;
  let iteration = 1;
  let iterator = await ds.iterator();

  // eslint-disable-next-line no-constant-condition
  while (true) {
    let next = await iterator.next();
    if (next.done) {
      epoch++;
      if (config.epochs && epoch > config.epochs) {
        break;
      }
      iterator = await ds.iterator();
      next = await iterator.next();
    }
    const { x, y } = next.value;

    // Keep loss for reporting
    let loss: tf.Tensor<tf.Rank> = null as any;
    const optFunc = (): tf.Scalar => {
      const logits = model.apply(x);
      loss = tf.keep(tf.losses.softmaxCrossEntropy(y, logits));
      return loss.asScalar();
    };
    tf.tidy(() => {
      const { grads } = opt.computeGradients(optFunc);
      const gradsClipped = clipByGlobalNormObj(grads, 1);
      opt.applyGradients(gradsClipped);
    });

    const lossVal = await loss.array();
    if (Array.isArray(config.callbacks)) {
      for (const callback of config.callbacks) {
        await callback(model, lossVal, iteration);
      }
    }

    // Dispose everything
    loss.dispose();
    x.dispose();
    y.dispose();

    // Check if we should stop
    iteration++;
    if (config.maxIter && iteration > config.maxIter) {
      break;
    }

    if (config.verbose) {
      console.log("Mem:", tf.memory());
      console.log(`Epoch: ${epoch}, Step: ${iteration}, Loss: ${lossVal}`);
    }

    await new Promise((resolve) => setTimeout(resolve, 1));
  }
}
