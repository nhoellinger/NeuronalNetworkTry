package neuronalnetworkapi.layer;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;

public class ConvolutionalLayer extends Layer3D{
    
    double[][][][] filters;

    public ConvolutionalLayer(int inputDepth, int inputLen, int stride, int padding, int nrFilters, int filterSize) {
        super(inputDepth, inputLen, filterSize, stride, padding, nrFilters);
        activation = (in, out) -> relu(in, out);
        dActivation = (in, out) -> dRelu(in, out);
        filters = new double[nrFilters][inputDepth][filterSize][filterSize];
        for(int i = 0; i < nrFilters; i++) {
            for(int j = 0; j < inputDepth; j++) {
                for(int m = 0; m < filterSize; m++) {
                    for(int n = 0; n < filterSize; n++) {
                        // init randomly with number between -0.01 and +0.01
                        filters[i][j][m][n] = (Math.random() - 0.5) * 0.02;
                    }
                }
            }
        }
    }
    
    @Override
    public double[] compute(double[] input) {
        for(int i = 0; i < outputDepth; i++) {
            for(int j = 0; j < outputLen; j++) {
                for(int k = 0; k < outputLen; k++) {
                    double val = 0.0;
                    for(int l = 0; l < inputDepth; l++) {
                        for(int m = 0; m < kernelSize; m++) {
                            for(int n = 0; n < kernelSize; n++) {
                                val += input[l * inputLen * inputLen + (j + m) * inputLen + k + n] * filters[i][l][m][n];
                            }
                        }
                    }
                    z[i * outputLen * outputLen + j * outputLen + k] = val;
                }
            }
        }
        activation.accept(z, a);
        return a;
    }

    @Override
    public void adaptWeights() {
        int outputLenSquare = outputLen * outputLen;
        for(int i = 0; i < outputDepth; i++) {
            for(int j = 0; j < inputDepth; j++) {
                for(int k = 0; k < outputLenSquare; k++) {
                    for(int m = 0; m < kernelSize; m++) {
                        for(int n = 0; n < kernelSize; n++) {
                            filters[i][j][m][n] -= learningRate * delta[i * outputLenSquare + k];
                        }
                    }
                }
            }
        }
    }

    @Override
    protected void backPropagatePrevLayer(double[] dz, double[] delta) {
        Arrays.fill(delta, 0);
        for(int i = 0; i < outputDepth; i++) {
            for(int j = 0; j < inputDepth; j++) {
                double[][] convDelta = fullConvolve(deltaTo2D(this.delta, i), i, j);
                for(int m = 0; m < inputLen; m++) {
                    for(int n = 0; n < inputLen; n++) {
                        delta[j * inputLen * inputLen + m * inputLen + n] += convDelta[inputLen - 1 - m][inputLen - 1 - n];
                    }
                }
            }
        }
        for(int i = 0; i < inputSize; i++) {
            delta[i] *= dz[i];
        }
    }

    @Override
    public void print(PrintWriter writer) throws IOException{
        for(double[][][] filt3d : filters) {
            for(double[][] filt2d : filt3d) {
                for(double[] filt1d : filt2d) {
                    for(double val : filt1d) {
                        writer.print(val);
                        writer.print("   ");
                    }
                    writer.println();
                }
                writer.println();
            }
            writer.println();
        }
        writer.flush();
    }
    
    private double[][] deltaTo2D(double[] delta, int index) {
        double[][] result = new double[outputLen][outputLen];
        for(int i = 0; i < outputLen; i++) {
            System.arraycopy(delta, index  * outputLen * outputLen + i * outputLen, result[i], 0, outputLen);
        }
        return result;
    }
    
    private double[][] fullConvolve(double[][] input, int f, int d) {
        double[][] res = new double[inputLen][inputLen];
        double[][] paddedInput = pad(input, kernelSize - 1, kernelSize - 1);
        double[][] flippedFilter = flip(filters[f][d]);
        for(int i = 0; i < inputLen; i++) {
            for(int j = 0; j < inputLen; j++) {
                double val = 0.0;
                for(int m = 0; m < kernelSize; m++) {
                    for(int n = 0; n < kernelSize; n++) {
                        val += paddedInput[i + m][j + n] * flippedFilter[m][n];
                    }
                }
                res[i][j] = val;
            }
        }
        return res;
    }
    
    private double[][] flip(double[][] input) {
        double[][] output = new double[input[0].length][input.length];
        for(int i = 0; i < output.length; i++) {
            for(int j = 0; j < output[i].length; j++) {
                output[i][j] = input[output[i].length - i - 1][output.length - j - 1];
            }
        }
        return output;
    }
    
    private double[][] pad(double[][] input, int horizontal, int vertical) {
        double[][] output = new double[input.length + 2 * vertical][input[0].length + 2 * horizontal];
        for(int i = 0; i < input.length; i++) {
            System.arraycopy(input[i], 0, output[i + vertical], horizontal, input[i].length);
        }
        return output;
    }
    
    private void relu(double[] in, double[] out) {
        for(int i = 0; i < out.length; i++) {
            out[i] = Math.max(0, in[i]);
        }
    }
    
    private void dRelu(double[] in, double[] out) {
        for(int i = 0; i < in.length; i++) {
            if(in[i] > 0) {
                out[i] = 1;
            } else {
                out[i] = 0;
            }
        }
    }
}
