import java.util.ArrayList;

public class Layer {
    private ArrayList<Neuron> neurons;

    public Layer(int numNeurons, int inputSize) {
        neurons = new ArrayList<>();
        for (int i = 0; i < numNeurons; i++) {
            // Initialize weights with small random values
            ArrayList<Double> weights = new ArrayList<>();
            for (int j = 0; j < inputSize; j++) {
                weights.add(Math.random() * 0.01);
            }
            neurons.add(new Neuron(weights, 0.0));
        }
    }
    public double[] forward(double[] inputs) {
        double[] outputs = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            outputs[i] = neurons.get(i).activate(inputs);
        }
        return outputs;
    }
    public ArrayList<Neuron> getNeurons() {
        return neurons;
    }
}