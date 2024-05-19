const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
    try {

        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        const prediction = model.predict(tensor);

        const score = await prediction.data();
        
        const classes = ['Non-Cancer', 'Cancer'];

        const classResult = score[0] > 0.5 ? 1 : 0; 
        const label = classes[classResult];

        
        const suggestions = {
            'Cancer': 'Segera periksa ke dokter.',
            'Non-Cancer': 'Senantiasa jaga kesehatan kulit Anda.'
        };

        
        const suggestion = suggestions[label];

        
        return {label, suggestion};
    } catch (error) {
        throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
    }
}

module.exports = predictClassification;
