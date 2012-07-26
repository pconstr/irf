try {
  module.exports = require('./build/Release/irf.node');
} catch(err) {
  module.exports = require('./build/Debug/irf.node');
}
