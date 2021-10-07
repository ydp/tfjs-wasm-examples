/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
const {nodeFileSystemRouter} = require('@tensorflow/tfjs-node/dist/io/file_system');

// Add the WASM backend to the global backend registry.
require('@tensorflow/tfjs-backend-wasm');
// Set the backend to WASM and wait for the module to be ready.
tf.setBackend('wasm').then(() => {

  const argparse = require('argparse');
  const https = require('https');
  const fs = require('fs');
  const createDataset = require('./data');
  
  /**
   * Builds and returns Multi Layer Perceptron Regression Model.
   *
   * @param {number} inputShape The input shape of the model.
   * @returns {tf.Sequential} The multi layer perceptron regression mode  l.
   */
  const createModel = function createModel(inputShape) {
    const model = tf.sequential();
    model.add(tf.layers.dense({
      inputShape: inputShape,
      activation: 'sigmoid',
      units: 50,
    }));
    model.add(tf.layers.dense({
      activation: 'sigmoid',
      units: 50,
    }));
    model.add(tf.layers.dense({
      units: 1,
    }));
    model.compile({optimizer: tf.train.sgd(0.01), loss: 'meanSquaredError'});
    return model;
  }
  
  const csvUrl =
      'https://storage.googleapis.com/tfjs-examples/abalone-node/abalone.csv';
  const csvPath = './abalone.csv';
  
  /**
   * Train a model with dataset, then save the model to a local folder.
   */
  async function run(epochs, batchSize, savePath) {
    const datasetObj = await createDataset('file://' + csvPath);
    const model = createModel([datasetObj.numOfColumns]);
    // The dataset has 4177 rows. Split them into 2 groups, one for training and
    // one for validation. Take about 3500 rows as train dataset, and the rest as
    // validation dataset.
    const trainBatches = Math.floor(3500 / batchSize);
    const dataset = datasetObj.dataset.shuffle(1000).batch(batchSize);
    const trainDataset = dataset.take(trainBatches);
    const validationDataset = dataset.skip(trainBatches);
    model.summary(); 
    console.log(`>>> backend: ` + tf.getBackend())
    await model.fitDataset(
        trainDataset, {
           epochs: epochs,
           validationData: validationDataset,
           callbacks: {
             onEpochEnd: async (epoch, logs) => {
               console.log(`Epoch: ${epoch} - loss: ${logs.loss.toFixed(3)}`);
             }
           }
       });
  
    tf.io.registerLoadRouter(nodeFileSystemRouter);
    tf.io.registerSaveRouter(nodeFileSystemRouter);
    console.log(`>>> backend: ` + tf.getBackend())
    await model.save(savePath);
    // console.log(model.getWeights())
    // for (let i = 0; i < model.getWeights().length; i++) {
    //  console.log(model.getWeights()[i].dataSync());
    // }
  
    console.log(`>>> backend: ` + tf.getBackend())
    const loadedModel = await tf.loadLayersModel(savePath + '/model.json');
    const result = model.predict(
        tf.tensor2d([[0, 0.625, 0.495, 0.165, 1.262, 0.507, 0.318, 0.39]]));
    console.log(
        'The actual test abalone age is 10, the inference result from the model is ' +
        result.dataSync());
  }
  
  const parser = new argparse.ArgumentParser(
      {description: 'TensorFlow.js-Node Abalone Example.', addHelp: true});
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 100,
    help: 'Number of epochs to train the model for.'
  });
  parser.addArgument('--batch_size', {
    type: 'int',
    defaultValue: 500,
    help: 'Batch size to be used during model training.'
  })
  parser.addArgument(
      '--savePath',
      {type: 'string', defaultValue: 'file:///tmp/trainedModel', help: 'Path.'})
  const args = parser.parseArgs();
  
  
  const file = fs.createWriteStream(csvPath);
  https.get(csvUrl, function(response) {
    response.pipe(file).on('close', async () => {
      run(args.epochs, args.batch_size, args.savePath);
    });
  });

});
