const path = require('path');
const webpack = require('webpack');
const nodeExternals = require('webpack-node-externals');

module.exports = {
  mode: 'none', //production
  entry: ['./src/main.ts'], // Entry point of your Nest.js application
  target: 'node',
  externals: [/node_modules/, nodeExternals()],
  output: {
    path: path.resolve(__dirname, 'dist'), // Output directory path
    filename: 'bundle.js', // Output file name
  },
  resolve: {
    extensions: ['.ts', '.filter.ts', '.exception.ts', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.node$/,
        loader: 'node-loader',
      },
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  plugins: [
    new webpack.IgnorePlugin({
      checkResource(resource) {
        const lazyImports = ['@nestjs/microservices', '@nestjs/platform-express', 'cache-manager', 'class-validator', 'class-transformer'];
        if (!lazyImports.includes(resource)) {
          return false;
        }
        try {
          require.resolve(resource);
        } catch (err) {
          return true;
        }
        return false;
      },
    }),
  ],
};
