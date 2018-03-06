package neuronalnetworkapi;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Paths;
import neuronalnetworkapi.layer.ConvolutionalLayer;
import neuronalnetworkapi.layer.FullyConnectedLayer;
import neuronalnetworkapi.layer.CrossEntropyOutputLayer;
import neuronalnetworkapi.layer.PoolingLayer;

public class ConvolutionalNeuronalNetwork {

    ConvolutionalLayer conv1 = new ConvolutionalLayer(1, 28, 1, 0, 12, 5);
    ConvolutionalLayer conv2 = new ConvolutionalLayer(12, 12, 1, 0, 16, 3);
    PoolingLayer pool1 = new PoolingLayer(12, 24, 2, 2);
    PoolingLayer pool2 = new PoolingLayer(16, 10, 2, 2);
    FullyConnectedLayer fc = new FullyConnectedLayer(400, 200);
    CrossEntropyOutputLayer out = new CrossEntropyOutputLayer(200, 62);
    
    public double[] train(double[] input, int correctOutputIndex) {
        double[] result = compute(input);
        backPropagate(correctOutputIndex);
        return result;
        
    }
    
    private void backPropagate(int correctOutputIndex) { 
        out.setCorrectIndex(correctOutputIndex);
        out.backPropagate(null);
        fc.backPropagate(out);
        pool2.backPropagate(fc);
        conv2.backPropagate(pool2);
        pool1.backPropagate(conv2);
        conv1.backPropagate(pool1);
    }
    
    public double[] compute(double[] input) {
        double[] result = conv1.compute(input);
        result = pool1.compute(result);
        result = conv2.compute(result);
        result = pool2.compute(result);
        result = fc.compute(result);
        result = out.compute(result);
        return result;
    }
    
    public void print(String location) throws FileNotFoundException, IOException {
        PrintWriter writer = new PrintWriter(Paths.get(location, "conv1.txt").toString());
        conv1.print(writer);
        writer.close();
        writer = new PrintWriter(Paths.get(location, "conv2.txt").toString());
        conv2.print(writer);
        writer.close();
        writer = new PrintWriter(Paths.get(location, "fc.txt").toString());
        fc.print(writer);
        writer.close();
        writer = new PrintWriter(Paths.get(location, "out.txt").toString());
        out.print(writer);
        writer.close();
    }
}
