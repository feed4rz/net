# CNN
Neural nets lab 3

## Installation
`npm install`

## Testing
Uncomment either runModel or initTraining at the bottom of the ./src/index.js and then run `npm run start`

## Models
There are two trained models:
- Digits (model_digits.json) based on EMNIST Digits
- Letters (model_letters.json) based on EMNIST Letters

EMNIST Dataset - https://www.kaggle.com/crawford/emnist

For now you can re-train and predict Letters dataset only (because it is kinda hardcoded :D)

![Running a model](https://i.imgur.com/m5nEtKP.png?2)

## class Net
### new Net({ layers, rate, iterations })

Options:
```json
{
  "layers": "Array of layers",
  "rate": "Learning rate from 0 to 1",
  "iterations": "Number of iterations on training set"
}
```

layers: Pass the array of layers with array elements being amount of neurons in a layer

Ex. [3, 4, 5]
 - 1-st layer: 3 neurons
 - 2-nd layer: 4 neurons
 - 3-rd layer: 5 neurons


Example:
```js
const net = new Net({
  layers: [784, 512, 26],
  rate: 0.3,
  iterations: 1
})
```

### await net.train({ set, log, save })
Starts training process

Options:
```json
{
  "set": "Array of training examples",
  "log": "Boolean, enable or disable debug logs",
  "save": "Boolean, to save or not to save trained model"
}
```

Training example should look like:
```json
{
  "inputs": "Array of input values for each input neuron",
  "targets": "Array of desired output values for each output neuron"
}
```

### net.accuracy({ set, labels, log })
Calculates accuracy based on a given sample

Options:
```json
{
	"set": "Array of testing examples",
	"labels": "Array of labels for outputs",
	"log": "Boolean, enable or disable debug logs"
}
```

Training example should look like:
```json
{
	"inputs": "Array of input values for each input neuron",
	"label": "Target result label"
}
```

### net.predict({ inputs, labels, normalize })
Runs feed-forward algorithms and returns labeled predictions

Options:
```json
{
  "inputs": "Array of input values for each input neuron",
  "labels": "Label for each output neuron",
  "normalize": "Boolean, normalize the prediction or return all output neurons"
}
```

### await net.initialize({ path })
Load trained model

Options:
```json
{
  "path": "Path to model file"
}
```
