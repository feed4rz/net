const { loadDataset, draw } = require('./utils')
const Net = require('./Net')

const net = new Net({
	layers: [784, 512, 26],
	rate: 0.3,
	iterations: 1
})

async function initTraining(datasetPath, testDatasetPath) {
	console.log('loading training set...')
	console.time('dataset load')
	const set = await loadDataset({ path: datasetPath, type: 1 })
	console.timeEnd('dataset load')

	net.train({ set })

	console.log('loading testing set...')
	const test = await loadDataset({ path: testDatasetPath, type: 1 })
	const labels = Array.from('abcdefghijklmnopqrstuvwxyz')

	net.accuracy({ set: test, labels })

	console.log('predicting random test sample...')
	const index = Math.floor(Math.random() * test.length)
	const { inputs, targets, label, labelNormalized } = test[index]
	const prediction = net.predict({ inputs, labels })
	console.log(prediction, labelNormalized)
	console.log(draw(inputs))
}

async function runModel(modelPath, testDatasetPath) {
	console.log('loading model...')
	await net.initialize({ path: modelPath })

	console.log('loading testing set...')
	const test = await loadDataset({ path: testDatasetPath, type: 1 })
	const labels = Array.from('abcdefghijklmnopqrstuvwxyz')

	net.accuracy({ set: test, labels })

	console.log('predicting random test sample...')
	const index = Math.floor(Math.random() * test.length)
	const { inputs, labelNormalized } = test[index]
	const prediction = net.predict({ inputs, labels })
	console.log(prediction, labelNormalized)
	console.log(draw(inputs))
}

// initTraining(`${__dirname}/../emnist-letters-test.csv`, `${__dirname}/../emnist-letters-test-3.csv`)
// runModel(`${__dirname}/../model_letters.json`, `${__dirname}/../emnist-letters-test-3.csv`)