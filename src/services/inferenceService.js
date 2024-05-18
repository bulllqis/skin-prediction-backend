const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
    try {
        // Decode the image, resize, and preprocess it
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        // Predict the classification
        const prediction = model.predict(tensor);

        // Convert the prediction tensor to an array
        const score = await prediction.data();
        // const confidenceScore = Math.max(...score) * 100;

        // Define classes
        const classes = ['Non-Cancer', 'Cancer'];

        // Determine the predicted class
        const classResult = score[0] > 0.5 ? 1 : 0; // If score > 0.5, classify as 'Cancer', otherwise 'Non-Cancer'
        const label = classes[classResult];

        // Define suggestions
        const suggestions = {
            'Cancer': 'Segera periksa ke dokter.',
            'Non-Cancer': 'Senantiasa jaga kesehatan kulit Anda.'
        };

        // Get the suggestion based on the predicted class
        const suggestion = suggestions[label];

        // Return the result with suggestion
        return {label, suggestion};
    } catch (error) {
        throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
    }
}

module.exports = predictClassification;
