import type {
  NamedTensor,
  NamedVariableMap,
} from "@tensorflow/tfjs-core/dist/tensor_types";
import * as tf from "@tensorflow/tfjs-node";

const ENGINE = tf.engine();

export function l2Loss(tensor: tf.Tensor) {
  return tf.div(tf.sum(tf.square(tensor)), 2);
}

export function globalNorm(tensors: tf.Tensor[]) {
  // https://github.com/tensorflow/tensorflow/blob/c256c071bb26e1e13b4666d1b3e229e110bc914a/tensorflow/python/ops/clip_ops.py#L242
  const halfSquaredNorms: tf.Tensor[] = [];
  tensors.forEach((tensor) => {
    halfSquaredNorms.push(l2Loss(tensor));
  });
  const halfSquaredNorm = tf.sum(tf.stack(halfSquaredNorms));
  const norm = tf.sqrt(
    tf.mul(halfSquaredNorm, tf.scalar(2.0, halfSquaredNorm.dtype)),
  );
  return norm;
}

export function clipByGlobalNorm(
  tensors: tf.Tensor[],
  clipNorm: number,
  useNorm?: "euclidean" | "fro" | tf.Tensor,
) {
  // https://github.com/kamalkraj/minGPT-TF/blob/master/mingpt/optimization.py
  // https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/clip_ops.py#L291-L382
  /*
    To perform the clipping, the values t_list[i] are set to:
        t_list[i] * clip_norm / max(global_norm, clip_norm)
    where:
        global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
    */
  useNorm = useNorm || globalNorm(tensors);
  const dtype = "string" === typeof useNorm ? undefined : useNorm.dtype;
  const scale = tf.mul(
    clipNorm,
    tf.minimum(
      tf.div(tf.scalar(1.0), useNorm),
      tf.div(tf.scalar(1.0, dtype), clipNorm),
    ),
  );
  const tensorsClipped: tf.Tensor[] = [];
  tensors.forEach((tensor) => {
    tensorsClipped.push(tf.clone(tf.mul(tensor, scale)));
  });
  return tensorsClipped;
}

export function clipByGlobalNormObj(
  tensorsObj: tf.NamedTensorMap,
  clipNorm: number,
  useNorm?: "euclidean" | "fro",
): NamedVariableMap {
  const varNames = Object.keys(tensorsObj);
  const tensorsArr = varNames.map((n) => tensorsObj[n]);
  const tensorsArrClipped = clipByGlobalNorm(tensorsArr, clipNorm, useNorm);
  const tensorsObjClipped: { [varName: string]: tf.Tensor } = {};
  tensorsArrClipped.forEach((t, ti) => {
    tensorsObjClipped[varNames[ti]] = t;
  });
  return tensorsObjClipped as NamedVariableMap;
}

export const AdamW = class extends tf.AdamOptimizer {
  weightDecayRate: number;
  includeInWeightDecay: string[];
  excludeFromWeightDecay: string[];
  gradientClipNorm: number;

  constructor({
    learningRate = 0.1,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-7,
    weightDecayRate = 0,
    includeInWeightDecay = [],
    excludeFromWeightDecay = [],
    gradientClipNorm = 1.0,
  }: {
    learningRate?: number;
    beta1?: number;
    beta2?: number;
    epsilon?: number;
    weightDecayRate?: number;
    includeInWeightDecay?: string[];
    excludeFromWeightDecay?: string[];
    gradientClipNorm?: number;
  }) {
    const p = {
      learningRate,
      beta1,
      beta2,
      epsilon,
      weightDecayRate,
      includeInWeightDecay,
      excludeFromWeightDecay,
      gradientClipNorm,
    };

    super(p.learningRate, p.beta1, p.beta2, p.epsilon);
    // this.learningRate = p.learningRate
    this.weightDecayRate = p.weightDecayRate;
    this.includeInWeightDecay = p.includeInWeightDecay;
    this.excludeFromWeightDecay = p.excludeFromWeightDecay;
    this.gradientClipNorm = p.gradientClipNorm;
  }
  applyGradients(variableGradients: NamedVariableMap | NamedTensor[]) {
    // log(variableGradients, typeof variableGradients)
    const varNames = Array.isArray(variableGradients)
      ? variableGradients.map((v) => v.name)
      : Object.keys(variableGradients);
    // log(this.learningRate, varNames)

    // const varValues = varNames.map(n => variableGradients[n])
    // const varValuesClipped = clipByGlobalNorm(varValues, 1)
    // varValuesClipped.forEach((v, i) => variableGradients[varNames[i]] = v)

    // Apply weight decay
    varNames.forEach((name) => {
      if (this.includeInWeightDecay.includes(name)) {
        const value = ENGINE.registeredVariables[name];
        const newValue = tf.sub(
          value,
          tf.mul(this.learningRate, tf.mul(value, this.weightDecayRate)),
        );
        value.assign(newValue);
      }
    });

    super.applyGradients(variableGradients);

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
};
