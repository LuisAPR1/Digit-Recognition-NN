import java.io.*;
import java.util.ArrayList;
import java.util.List;


public class NeuralNetwork {
    private ArrayList<Layer> layers;
    private double learningRate;

    public NeuralNetwork(ArrayList<Layer> layers) {
        this.layers = layers;
    }
    public void train(double[][] inputs, double[] targets, double mseThreshold, double learningRate) {
        this.learningRate = learningRate;
        int epoch = 0;
        List<Double> mseHistory = new ArrayList<>();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("mse_values.txt"))) {

            while (true) {
                double totalError = 0.0;

                for (int i = 0; i < inputs.length; i++) {
                    double[] input = inputs[i];
                    double target = targets[i];

                    // Forward pass
                    double output = forward(input);

                    // Compute error
                    double error = target - output;
                    totalError += 0.5 * Math.pow(error, 2);

                    // Backward pass
                    backward(target);

                    // Update weights and biases
                    updateWeights(input);
                }

                double mse = totalError / inputs.length;

                // Add MSE to history
                mseHistory.add(mse);

                // Write MSE value to file
                writer.write(" " + String.format("%.100f", mse).replace('.', ','));
                writer.newLine();

                epoch++;

                // Check for stopping conditions
                if (mse < mseThreshold) {
                    System.out.println("Treinamento convergiu na época: " + epoch);
                    break;
                }

                // Every 10 epochs, check if MSE has increased compared to 10 epochs ago
                if (epoch % 10 == 0 && epoch >= 10) {
                    double previousMSE = mseHistory.get(mseHistory.size() - 10); // MSE from 10 epochs ago
                    if (mse > previousMSE) {
                        System.out.println("O MSE aumentou na época: " + epoch + ". Interrompendo o treinamento.");
                        break;
                    }
                }
            }

            // Após o treinamento, salvar os pesos
            saveWeights("pesos.csv");

        } catch (IOException e) {
            System.err.println("Erro ao escrever valores de MSE no arquivo: " + e.getMessage());
        }
    }
    public double forward(double[] input) {
        double[] outputs = input;
        for (Layer layer : layers) {
            outputs = layer.forward(outputs);
        }
        return outputs[0]; // Assuming the output layer has one neuron
    }
    public void backward(double target) {
        // Calculate delta for output layer
        Layer outputLayer = layers.get(layers.size() - 1);
        Neuron outputNeuron = outputLayer.getNeurons().get(0);
        double output = outputNeuron.getOutput();
        double delta = (target - output) * outputNeuron.sigmoidDerivative();
        outputNeuron.setDelta(delta);

        // Backpropagate deltas through hidden layers
        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Layer nextLayer = layers.get(i + 1);
            for (int j = 0; j < currentLayer.getNeurons().size(); j++) {
                Neuron neuron = currentLayer.getNeurons().get(j);
                double sum = 0.0;
                for (Neuron nextNeuron : nextLayer.getNeurons()) {
                    sum += nextNeuron.getWeights().get(j) * nextNeuron.getDelta();
                }
                double neuronDelta = sum * neuron.sigmoidDerivative();
                neuron.setDelta(neuronDelta);
            }
        }
    }
    public void updateWeights(double[] input) {
        double[] inputs = input;

        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                // Update weights
                ArrayList<Double> weights = neuron.getWeights();
                for (int j = 0; j < weights.size(); j++) {
                    double oldWeight = weights.get(j);
                    double inputVal = inputs[j];
                    double newWeight = oldWeight + learningRate * neuron.getDelta() * inputVal;
                    weights.set(j, newWeight);
                }
                // Update bias
                neuron.setBias(neuron.getBias() + learningRate * neuron.getDelta());
            }
            // Prepare inputs for the next layer
            inputs = new double[layer.getNeurons().size()];
            for (int i = 0; i < layer.getNeurons().size(); i++) {
                inputs[i] = layer.getNeurons().get(i).getOutput();
            }
        }
    }
    public void test(double[][] inputs, double[] targets) {
        double erro = 0;
        System.out.println("\n\nTesting the network:");
        int correct = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double target = targets[i];
            double output = forward(input);
            System.out.println("Input " + (i + 1) + ": Expected: " + (int) target + ", Predicted: " + output);

            erro += Math.sqrt(Math.pow((target - output), 2));

            if ((output >= 0.5 ? 1 : 0) == (int) target) {
                correct++;

            } else {
                System.out.println("aaa" + (i + 1));
            }
        }
        erro = erro / inputs.length;
        double accuracy = (double) correct / inputs.length * 100;
        System.out.println("\nAccuracy: " + String.format("%.2f", accuracy) + "%");
        System.out.println("\n" + correct + " CORRETOS DE " + inputs.length);
        System.out.println("\n VALOR DO ERRO - " + erro);

    }
    public void mooshake(double[][] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double output = forward(input);
            double rounded = Math.round(output * 100.0) / 100.0;
            System.out.println(rounded);        }

    }
    public void setWeights(ArrayList<Double> weights) {
        int weightIndex = 0;

        // Itera sobre as camadas da rede neural
        for (Layer layer : layers) {
            // Itera sobre os neurônios de cada camada
            for (Neuron neuron : layer.getNeurons()) {
                // Itera sobre os pesos de cada neurônio
                ArrayList<Double> neuronWeights = neuron.getWeights();
                for (int i = 0; i < neuronWeights.size(); i++) {
                    // Atribui os pesos do ArrayList para os pesos do neurônio
                    neuronWeights.set(i, weights.get(weightIndex));
                    weightIndex++;
                }
            }
        }
    }
    public void saveWeights(String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (Layer layer : layers) {
                for (Neuron neuron : layer.getNeurons()) {
                    // Salvar pesos
                    for (double weight : neuron.getWeights()) {
                        writer.write(weight + ",");
                    }
                    // Salvar bias
                    writer.write(neuron.getBias() + "");
                    writer.newLine();
                }
            }
            System.out.println("Pesos salvos em " + filename);
        } catch (IOException e) {
            System.err.println("Erro ao salvar os pesos: " + e.getMessage());
        }
    }
    public void loadWeights(String filename) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            int layerIndex = 0;
            int neuronIndex = 0;

            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");
                Layer layer = layers.get(layerIndex);
                Neuron neuron = layer.getNeurons().get(neuronIndex);

                ArrayList<Double> weights = new ArrayList<>();
                for (int i = 0; i < tokens.length - 1; i++) {
                    weights.add(Double.parseDouble(tokens[i]));
                }
                double bias = Double.parseDouble(tokens[tokens.length - 1]);

                neuron.setWeights(weights);
                neuron.setBias(bias);

                neuronIndex++;
                if (neuronIndex >= layer.getNeurons().size()) {
                    neuronIndex = 0;
                    layerIndex++;
                    if (layerIndex >= layers.size()) {
                        break;
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Erro ao carregar os pesos: " + e.getMessage());
        }
    }
}