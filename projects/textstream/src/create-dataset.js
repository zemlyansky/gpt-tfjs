const tf = require('@tensorflow/tfjs')

const ReadStream = require('filestream').read
const { encode, decode } = require('gpt-tokenizer/encoding/r50k_base')
const vocabSize = 50257

async function sleep (t) {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, t);
  });
}

function createDatasetFromTextStreams(getStreams, blockSize) {
    let runs = 0
    let steps = 0

    async function* dataGenerator() {
        // This is restarted on each epoch
        // Because streams can only be read once, we need to reinit them here
        // That's why we use a function to get the streams, rather than the streams themselves
        // In nodejs it will return streams from fs,
        // In the browser it will return streams from a file input
        const streams = getStreams()
        if (!Array.isArray(streams)) {
            streams = [streams]
        }
        console.log('Starting data generator:', runs++)
        for await (const stream of streams) {
            for await (const chunk of stream) {
                const text = chunk.toString()
                const tokens = encode(text)
                console.log(`Stream chunk: ${text.slice(0, 40)}... (${tokens.length} tokens)`)
                for (let i = 0; i < tokens.length - blockSize - 1; i += blockSize) {
                    const x = tokens.slice(i, i + blockSize)
                    const y = tokens.slice(i + 1, i + blockSize + 1)
                    yield {
                        x,
                        y
                    }
                    steps++
                    await sleep(1)
                }
                await sleep(1)
            }
        }
    }

    return tf.data.generator(dataGenerator)
        .map(v => ({
            x: tf.cast(v.x, 'int32'), 
            y: tf.oneHot(tf.cast(v.y, 'int32'), vocabSize)
        }))
}

function createDatasetFromFileList (fileList, blockSize, chunkSize=256) {
    const files = Array.from(fileList)
    function getStreams () {
        return files.map(file => new ReadStream(file, { chunkSize: chunkSize }))
    }
    return createDatasetFromTextStreams(getStreams, blockSize)
}

module.exports = {
    createDatasetFromTextStreams,
    createDatasetFromFileList,
    vocabSize
}