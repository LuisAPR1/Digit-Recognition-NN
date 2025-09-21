import java.util.ArrayList;

public class Neuron {

    private ArrayList<Double> weights;
    private double bias;
    private double output;
    private double delta;

    public Neuron(ArrayList<Double> weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public double netInput(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < weights.size(); i++) {
            sum += weights.get(i) * inputs[i];
        }
        return sum;
    }

    public double activate(double[] inputs) {
        double netInput = netInput(inputs);
        output = sigmoid(netInput);
        return output;
    }

    private double sigmoid(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    public double sigmoidDerivative() {
        return output * (1 - output);
    }

    public ArrayList<Double> getWeights() {
        return weights;
    }

    public void setWeights(ArrayList<Double> weights) {
        this.weights = weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getOutput() {
        return output;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }
}