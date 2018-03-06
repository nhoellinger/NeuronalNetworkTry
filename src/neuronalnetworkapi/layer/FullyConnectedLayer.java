package neuronalnetworkapi.layer;

import java.io.IOException;
import java.io.PrintWriter;

public class FullyConnectedLayer extends Layer{
    
    double[][] weights; // double[input][output]
    
    public FullyConnectedLayer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
        weights = new double[inputSize][outputSize];
        for(int i = 0; i < inputSize; i++) {
            for(int j = 0; j < outputSize; j++) {
                weights[i][j] = (Math.random() - 0.5) * 0.02;
            }
        }
        activation = (in, out) -> sigmoid(in, out);
        dActivation = (in, out) -> dSigmoid(in, out);
    }
    
    @Override
    public double[] compute(double[] input) {
        for(int i = 0; i < outputSize; i++) {
            double result = 0.0;
            for(int j = 0; j < inputSize; j++) {
                result += input[j] * weights[j][i];
            }
            z[i] = result;
        }
        activation.accept(z, a);
        return a;
    }
    
    @Override
    public void adaptWeights() {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] -= learningRate * delta[j];
            }
        }
    }

    @Override
    public void print(PrintWriter writer) throws IOException {
        for(double[] weightsInput : weights) {
            for(double val : weightsInput) {
                writer.println(val);
            }
            writer.println();
        }
        writer.flush();
    }
    
    private void sigmoid(double[] input, double[] output) {
        for(int i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + Math.exp(-input[i]));
        }
    }
    
    private void dSigmoid(double[] in, double[] out) {
        for(int i = 0; i < in.length; i++) {
            double exp = Math.exp(-in[i]);
            double divisor = exp + 1;
            out[i] = exp / (divisor * divisor); 
        }
    }

    @Override
    public void backPropagatePrevLayer(double[] dz, double[] delta) {
        for(int i = 0; i < inputSize; i++) {
            double result = 0.0;
            for(int j = 0; j < outputSize; j++) {
                result += this.delta[j] * weights[i][j];
            }
            delta[i] = result * dz[i];
        }
    }
}
