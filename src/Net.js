const Matrix = require('node-matrix')
const fs = require('fs').promises

// Neural Net Class
class Net {
	/*
		Class constructor

		Options:
		{
			layers: Array of layers,
			rate: Learning rate from 0 to 1,
			iterations: Number of iterations on training set
		}

		layers:
		Pass the array of layers
		with array elements being
		amount of neurons in a layer
		Ex. [3, 4, 5]
		1-st layer: 3 neurons
		2-nd layer: 4 neurons
		3-rd layer: 5 neurons
	*/
	constructor(args = {}) {
		const { layers = [784, 500, 26], rate = 0.1, iterations = 5 } = args
		// Require at least 1 hidden layer
		if (layers.length < 2) {
			throw new Error('Please specify at least 2 layers')
		}

		this.w = [] // Array of weights matrixes
		this.l = layers
		this.i = layers[0] // Amount of inputs
		this.o = layers[layers.length - 1] // Amount of outputs
		this.r = rate
		this.it = iterations

		this.randomWeights()
	}

	/*
		Load trained model

		Options:
		{
			path: Path to model file
		}
	*/
	async initialize(args) {
		const { path } = args
		if (!path) {
			throw new Error('Please provide model path')
		}

		const model = JSON.parse(await fs.readFile(path))

		this.w = []

		for (let i = 0; i < model.length; i++) {
			const matrix = []
			for (let j = 0; j < model[i].numRows; j++) {
				matrix.push(model[i][j])
			}

			const m = Matrix(matrix)

			this.w.push(m)
		}
	}

	// Generate random weights
	randomWeights() {
		for (let i = 0; i < this.l.length - 1; i++) {
			// Weights between current and next layers
			const columns = this.l[i]
			const rows = this.l[i + 1]
			const values = () => this._randomValue(columns)

			const m = Matrix({
				rows,
				columns,
				values
			})

			this.w.push(m)
		}
	}

	// Used to initialize a weight
	_randomValue(neurons) {
		const spread = Math.pow(neurons, -.5)
		return Math.random() * 3 * spread - spread
	}

	static sigmoid(x) {
		return 1 / (1 + Math.pow(Math.E, -x))
	}

	// Sigmoid for activation
	_activate(x) {
		return Net.sigmoid(x)
	}

	/*
		Feed-forward algorithm

		Options:
		{
			inputs: Array of input values for each input neuron
		}
	*/
	_forward(args = {}) {
		const { inputs = [] } = args
		if (this.i !== inputs.length) {
			throw new Error('Invalid inputs object')
		}

		const inputsMatrix = Matrix(inputs.map(input => [input] ))
		const outputs = [inputsMatrix] // neuron outputs for each layer
		const outputsSpecial = [0] // output * (1 - output) for hidden and output layers

		// Propagate forward
		for (let i = 0; i < this.w.length; i++) {
			// activate(W * O)
			const output = Matrix.multiply(this.w[i], outputs[i]).transform(this._activate)

			// O * (1 - O)
			const outputSpecial = output.transform(num => num * (1 - num))

			outputs.push(output)
			outputsSpecial.push(outputSpecial)
		}

		return { outputs, outputsSpecial }
	}
	
	/*
		Runs a feed-forward and backward propagation algorithms

		Options:
		{
			inputs: Array of input values for each input neuron,
			targets: Array of desired output values for each output neuron,
			i: Dataset index,
			j: Iteration index,
			log: Boolean, enable or disable debug logs
		}
	*/
	_it(args = {}) {
		const { inputs = [], targets = [], i = 0, j = 0, log = true } = args
		if (this.i !== inputs.length) {
			throw new Error('Invalid inputs object')
		}
		if (this.o !== targets.length) {
			throw new Error('Invalid targets object')
		}

		// Propagate forward
		const { outputs, outputsSpecial } = this._forward({ inputs })

		// console.log(outputs, outputsSpecial)

		// Calculate errors
		const output = outputs[outputs.length - 1]
		const targetsMatrix = Matrix(targets.map(target => [target]))
		const errorOutput = Matrix.subtract(targetsMatrix, output)
		const errors = [errorOutput]

		let averageError = 0

		for (let i = 0; i < errorOutput.numRows; i++) {
			averageError = averageError + errorOutput[i].reduce((a, b) => (a * a) + (b * b), 0)
		}

		averageError = (averageError / (errorOutput.numRows * errorOutput.numCols)).toFixed(8)

		for (let i = this.w.length - 1; i > -1; i--) {
			const w = this.w[i]
			const e = errors[0]

			// Wj^T * Ek
			const error = Matrix.multiply(w.transpose(), e)

			errors.unshift(error)
		}

		// console.log(errors)

		// Propagate backward
		for (let j = this.w.length - 1; j > -1; j--) {
			// deltaWj = a * Ek * Ok(1 - Ok) * Oj^T
			const k = j + 1
			const delta = Matrix.multiplyScalar(
				Matrix.multiply(
					Matrix.multiplyElements(errors[k], outputsSpecial[k]),
					outputs[j].transpose()
				),
				this.r
			)
			
			// Wj = Wj + deltaWj
			this.w[j] = Matrix.add(this.w[j], delta)
			// this.w[j] = this.w[j].transform(this.generator)
		}

		if (log) {
			console.log(`iteration ${j}:${i}: ${averageError}`)
		}
	}

	/*
		Starts training process

		Options:
		{
			set: Array of training examples,
			log: Boolean, enable or disable debug logs,
			save: Boolean, to save or not to save trained model
		}

		Training example should look like:
		{
			inputs: Array of input values for each input neuron,
			targets: Array of desired output values for each output neuron
		}
	*/
	async train(args = {}) {
		const { set = [], log = true, save = true } = args
		if (!set || !set.length) {
			throw new Error('Invalid set object')
		}
		
		const startMs = Date.now()
		if (log) {
			console.log('training was started...')
		}

		for (let j = 0; j < this.it; j++) {
			for (let i = 0; i < set.length || i < this.it - 1; i++) {
				this._it({ ...set[i], i, j, log })
			}
		}

		const filename = `model_${Date.now()}.json`
		if (save) {
			await fs.writeFile(filename, JSON.stringify(this.w, null, 2))
		}

		const completedMs = Date.now() - startMs

		if (log) {
			console.log(`training was completed in ${completedMs}ms!\nmodel filename: ${filename}`)
		}
	}

	/*
		Calculates accuracy based on a given sample

		Options:
		{
			set: Array of testing examples,
			labels: Array of labels for outputs,
			log: Boolean, enable or disable debug logs
		}

		Training example should look like:
		{
			inputs: Array of input values for each input neuron,
			label: Target result label
		}
	*/
	accuracy(args = {}) {
		const { set = [], labels = [], log = true } = args
		if (!set || !set.length) {
			throw new Error('Invalid set object')
		}
		if (this.o !== labels.length) {
			throw new Error('Invalid labels object')
		}

		const startMs = Date.now()
		if (log) {
			console.log('testing was started...')
		}

		let accuracy = 0
		for (let i = 0; i < set.length; i++) {
			const { inputs, labelNormalized } = set[i]
			const { prediction } = this.predict({ inputs, labels })

			if (labelNormalized === prediction) {
				accuracy++
			}
		}

		accuracy = Math.floor((accuracy / set.length) * 1000000) / 10000

		const completedMs = Date.now() - startMs

		if (log) {
			console.log(`testing was completed in ${completedMs}ms!\naccuracy: ${accuracy}%`)
		}

		return {
			accuracy
		}
	}

	/*
		Runs feed-forward algorithms and returns labeled predictions

		Options:
		{
			inputs: Array of input values for each input neuron,
			labels: Label for each output neuron,
			normalize: Boolean, normalize the prediction or return all output neurons
		}
	*/
	predict(args = {}) {
		const { inputs = [], labels = [], normalize = true } = args
		if (this.i !== inputs.length) {
			throw new Error('Invalid inputs object')
		}
		if (this.o !== labels.length) {
			throw new Error('Invalid labels object')
		}

		const { outputs } = this._forward({ inputs })
		const output = outputs[outputs.length - 1] // Last output layer

		if (!normalize) {
			// Labeled results
			const results = {}

			for (let i = 0; i < output.numRows; i++) {
				const label = labels[i]
				results[label] = output[i]
			}

			return results
		}

		let maxValue = 0
		let maxLabel = ''

		for (let i = 0; i < output.numRows; i++) {
			if (maxValue < output[i][0]) {
				maxValue = output[i][0]
				maxLabel = labels[i]
			}
		}

		return {
			probability: Math.floor(maxValue * 1000000) / 10000,
			prediction: maxLabel
		}
	}
}

module.exports = Net