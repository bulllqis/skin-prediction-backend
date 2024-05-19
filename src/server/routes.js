const { postPredictHandler, getPredictionHistories } = require('./handler');
 
const routes = [
  {
    path: '/predict',
    method: 'POST',
    handler: postPredictHandler,
    options: {
      payload: {
        maxBytes: 1000000, 
        allow: 'multipart/form-data',
        multipart: true
      }
    }
  },

  {
    path: '/predict/histories',
    method: 'GET',
    handler: getPredictionHistories
  }
]
 
module.exports = routes;