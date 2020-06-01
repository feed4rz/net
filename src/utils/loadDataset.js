const parse = require('csv-parse/lib/sync')
const fs = require('fs').promises

async function loadDataset(args = {}) {
	const { path = `${__dirname}/emnist-letters-test.csv`, type = 0 } = args

	const content = await fs.readFile(path)
	const records = parse(content)

	const set = []
	const alphabet = type ? 'abcdefghijklmnopqrstuvwxyz' : '0123456789'

	records.map(record => {
		const label = +record[0]
		const labelNormalized = alphabet[label - type]
		record.splice(0, 1)
		const inputs = record.map(input => +input / 255)
		const targets = Array.from(alphabet).map((letter, i) => {
			return i === label - type ? .99 : .01
		})

		set.push({
			inputs,
			label,
			labelNormalized,
			targets
		})
	})

	return set
}

module.exports = loadDataset