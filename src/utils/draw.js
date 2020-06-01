function draw(inputs) {
	let result = ''

	for (let i = 0; i < 28; i++) {
		for (let j = 0; j < inputs.length; j += 28) {
			const input = inputs[i + j]
			const char = getChar(input)

			result += char + char
		}

		result += '\n'
	}

	return result
}

function getChar(input) {
	const chars = '█▓▒░ '

	if (input < .2) {
		return chars[4]
	}
	if (input < .4) {
		return chars[3]
	}
	if (input < .6) {
		return chars[2]
	}
	if (input < .8) {
		return chars[1]
	}
	if (input <= 1) {
		return chars[0]
	}
}

module.exports = draw