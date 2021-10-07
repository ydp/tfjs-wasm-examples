/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-wasm');
tf.setBackend('wasm').then(() => {
  const argparse = require('argparse');
  
  const data = require('./data');

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({rate: 0.25}));
  model.add(tf.layers.dense({units: 512, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.5}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  
  const optimizer = 'rmsprop';
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  
  async function run(epochs, batchSize, modelSavePath) {
    await data.loadData();
  
    const {images: trainImages, labels: trainLabels} = data.getTrainData();
    model.summary();
  
    let epochBeginTime;
    let millisPerStep;
    const validationSplit = 0.15;
    const numTrainExamplesPerEpoch =
        trainImages.shape[0] * (1 - validationSplit);
    const numTrainBatchesPerEpoch =
        Math.ceil(numTrainExamplesPerEpoch / batchSize);
    await model.fit(trainImages, trainLabels, {
      epochs,
      batchSize,
      validationSplit
    });
  
    const {images: testImages, labels: testLabels} = data.getTestData();
    const evalOutput = model.evaluate(testImages, testLabels);
  
    console.log(
        `\nEvaluation result:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);
  
    if (modelSavePath != null) {
      await model.save(`file://${modelSavePath}`);
      console.log(`Saved model to path: ${modelSavePath}`);
    }
  }
  
  const parser = new argparse.ArgumentParser({
    description: 'TensorFlow.js-Node MNIST Example.',
    addHelp: true
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 20,
    help: 'Number of epochs to train the model for.'
  });
  parser.addArgument('--batch_size', {
    type: 'int',
    defaultValue: 128,
    help: 'Batch size to be used during model training.'
  })
  parser.addArgument('--model_save_path', {
    type: 'string',
    help: 'Path to which the model will be saved after training.'
  });
  const args = parser.parseArgs();
  
  run(args.epochs, args.batch_size, args.model_save_path);
});
