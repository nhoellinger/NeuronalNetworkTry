package neuronalnetworkapi.layer;

public class CrossEntropyOutputLayer extends FullyConnectedLayer {
    
    int correctIndex;
    
    public CrossEntropyOutputLayer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
        activation = (in, out) -> softMax(in, out);
    }
    
    public void setCorrectIndex(int correctIndex) {
        this.correctIndex = correctIndex;
    }
    
    @Override
    public void backPropagate(Layer nextLayer) {
        calcDelta();
        adaptWeights();
    }
    
    // multi-class cross entropy loss: l_i = -y_i * ln(a_i)
    // therefore d_i = l_i' % softmax'(a_i) = a_i - y_i;
    private void calcDelta() {
        System.arraycopy(a, 0, delta, 0, a.length);
        delta[correctIndex] -= 1;
    }
    
    private void softMax(double[] input, double[] output) {
        double total = 0.0;
        for(int i = 0; i < output.length; i++) {
            double out = Math.exp(input[i]);
            output[i] = out;
            total += out;
        }
        for(int i = 0; i < output.length; i++) {
            output[i] /= total;
        }
    }
}
