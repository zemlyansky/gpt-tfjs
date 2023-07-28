import * as tf from "@tensorflow/tfjs-node";

export function convertMinGPTConfig(config) {
  const mapping = {
    n_embd: "nEmbd",
    n_head: "nHead",
    n_layer: "nLayer",
    block_size: "blockSize",
    vocab_size: "vocabSize",
    attn_pdrop: "dropout",
    resid_pdrop: "dropout",
    embd_pdrop: "dropout",
    model_type: "modelType",
  };
  const newConfig = {};
  for (const key in config) {
    if (key in mapping) {
      newConfig[mapping[key]] = config[key];
    } else {
      newConfig[key] = config[key];
    }
  }
  return newConfig;
}

export function convertMinGPTWeights(weights) {
  const newWeights: { [key: string]: tf.Tensor } = {};
  for (const wn in weights) {
    const w = weights[wn];
    let wt = tf.tensor(w);
    // Prepare names
    let wnNew = wn.replace(/\./g, "/");
    if (wnNew.includes("ln_")) {
      wnNew = wnNew.replace("weight", "gamma");
      wnNew = wnNew.replace("bias", "beta");
    } else if (wnNew.includes("wte") || wnNew.includes("wpe")) {
      wnNew = wnNew.replace("weight", "embeddings");
    } else {
      wnNew = wnNew.replace("weight", "kernel");
      wnNew = wnNew.replace("bias", "bias");
    }
    if (wnNew.includes("kernel") && wt.shape.length == 2) {
      wt = wt.transpose();
    }
    newWeights[wnNew] = wt;
  }
  return newWeights;
}
