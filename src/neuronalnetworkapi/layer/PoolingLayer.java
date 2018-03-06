package neuronalnetworkapi.layer;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

public class PoolingLayer extends Layer3D{
    
    int[] poolingIndex;

    public PoolingLayer(int inputDepth, int inputLen, int stride, int poolingSize) {
        super(inputDepth, inputLen, poolingSize, stride, 0, inputDepth);
        poolingIndex = new int[outputSize];
        activation = (in, out) -> System.arraycopy(in, 0, out, 0, outputSize);
        dActivation = (in, out) -> System.arraycopy(in, 0, out, 0, outputSize);
    }
    
    @Override
    public double[] compute(double[] input) {
        int outputIndex = 0;
        for(int i = 0; i < inputDepth; i++) {
            for(int j = 0; j < inputLen; j+=stride) {
                for(int k = 0; k < inputLen; k+=stride) {
                    double max = Double.NEGATIVE_INFINITY;
                    int maxInputIndex = 0;
                    for(int m = 0; m < kernelSize; m++) {
                        for(int n = 0; n < kernelSize; n++) {
                            int inputIndex = i * inputLen * inputLen + (j + m) * inputLen + k + n;
                            double val = input[inputIndex];
                            if(val > max) {
                                max = val;
                                maxInputIndex = inputIndex;
                            }
                        }
                    }
                    poolingIndex[outputIndex] = maxInputIndex;
                    z[outputIndex++] = max;
                }
            }
        }
        activation.accept(z, a);
        return a;
    }

    @Override
    public void adaptWeights() {
    }

    @Override
    protected void backPropagatePrevLayer(double[] dz, double[] delta) {
        Arrays.fill(delta, 0);
        for(int i = 0; i < outputSize; i++) {
            int index = poolingIndex[i];
            delta[index] = this.delta[i] * dz[index];
        }
    }

    @Override
    public void print(PrintWriter writer) throws IOException {
    }
}
