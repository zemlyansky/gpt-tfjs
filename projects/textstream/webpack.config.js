const path = require('path')
const webpack = require('webpack')
const package = require('./package.json')

module.exports = (env) => {
  const config = {
    entry: './src/train-browser.js',
    output: {
      filename: 'bundle.js',
      path: path.resolve(__dirname, 'dist'),
    },
    plugins: [
      new webpack.ProvidePlugin({
        Buffer: ['buffer', 'Buffer'],
      }),
      new webpack.ProvidePlugin({
        process: 'process/browser',
      }),
    ],
    // Ignore tfjs
    externals: {
      '@tensorflow/tfjs': 'tf',
    },
    // Load different versions of vue based on RUNTIME value
    resolve: {
      fallback: {
        'buffer': require.resolve('buffer'),
        'path': require.resolve('path-browserify'),
        'stream': require.resolve('stream-browserify'), // Needed for csv-parse
      }
    }
  }

  return config
}
