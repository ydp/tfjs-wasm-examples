# tfjs-wasm-examples
tensorflow.js wasm backend examples

Supported op:

[TFJS Ops Matrix](https://docs.google.com/spreadsheets/d/1D25XtWaBrmUEErbGQB0QmNhH-xtwHo9LDl59w0TbxrI/edit#gid=0)

Supported op in code:

[register_all_kernels](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/src/register_all_kernels.ts)

## Install node.js and yarn

```
### Ubuntu
curl -fsSL https://deb.nodesource.com/setup_current.x | bash -
apt install -y nodejs
 
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list

apt update && apt install -y yarn

### Mac
# wget https://nodejs.org/dist/v16.10.0/node-v16.10.0.pkg
# curl -o- -L https://yarnpkg.com/install.sh | bash
brew install nodejs
brew install yarn
```

## Install packages

```
yarn
```

## Models

 * mnist
 * abalone
 * baseball
 * mobilenet


## ImageNet sample images:

 https://github.com/EliSchwartz/imagenet-sample-images

