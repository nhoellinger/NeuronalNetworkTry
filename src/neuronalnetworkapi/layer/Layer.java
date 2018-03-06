package neuronalnetworkapi.layer;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.function.BiConsumer;

public abstract class Layer implements ILayer{
    int inputSize;
    int outputSize;
    double learningRate = 0.2;
    double[] a; // net input after activation (= output)
    double[] z; // net input (= weighted sums of inputs)
    double[] dz;
    double[] delta;
    double[] batchDelta;
    BiConsumer<double[], double[]> activation;
    BiConsumer<double[], double[]> dActivation;

    public Layer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        a = new double[outputSize];
        z = new double[outputSize];
        dz = new double[outputSize];
        delta = new double[outputSize];
        batchDelta = new double[outputSize];
    }
    
    @Override
    public void backPropagate(Layer nextLayer) {
        dActivation.accept(z, dz);
        nextLayer.backPropagatePrevLayer(dz, delta);
        addDelta();
        adaptWeights();
    }
    
    protected void addDelta() {
        for(int i = 0; i < outputSize; i++) {
            batchDelta[i] += delta[i];
        }
    }
    
    public abstract void adaptWeights();
    
    // delta = out parameter
    protected abstract void backPropagatePrevLayer(double[] dz, double[] delta);
    
    public abstract void print(PrintWriter writer) throws IOException;
}
