// Import @tensorflow/tfjs or @tensorflow/tfjs-core
//const tf = require('@tensorflow/tfjs');
import * as tf from '@tensorflow/tfjs';
// Adds the WASM backend to the global backend registry.
import '@tensorflow/tfjs-backend-wasm';
import { get, getSync } from '@andreekeberg/imagedata'
import {IMAGENET_CLASSES} from './imagenet_classes.mjs';

const image_name = './n01443537_goldfish.JPEG'

async function main() {
  let img = getSync(image_name)
  console.log('>>> predict input:')
  console.log(img)
  const tensor = tf.browser.fromPixels({data: new Uint8Array(img.data), width: img.width, height: img.height})
    .resizeBilinear([224, 224])
    .cast('float32')
    .div(255)
    .expandDims(0)
    .toFloat();
  let model = await tf.loadGraphModel(
    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2',
    {fromTFHub: true});

  const logits = model.predict(tensor);

  console.log('>>> predict output:');
  logits.print();
  console.log('output shape:', logits.shape);
  // console.log('logits after squeeze:', logits.squeeze().shape)

  const softmax = tf.softmax(logits);
  const values = await softmax.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topK = 3;
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      index: topkIndices[i]-1,
      className: IMAGENET_CLASSES[topkIndices[i]-1],
      probability: topkValues[i]
    });
  }
  console.log(topClassesAndProbs);
}

// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm').then(() => main());
