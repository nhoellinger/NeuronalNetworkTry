package neuronalnetworkapi;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;

public class NeuronalNetworkApi {

    static TestCase[] testCases;
    
    public static void main(String[] args) throws IOException {
        readTrainFile();
        ConvolutionalNeuronalNetwork cnn = new ConvolutionalNeuronalNetwork();
        int i = 0;
        for(TestCase testCase : testCases) {
            if(i == 5000) {
                System.out.println("Debug!!!");
            }
            double[] result = cnn.train(toDoubleArray(testCase.data), testCase.label);
            System.out.println("Index: " + i++);
            System.out.println("Correct: " + testCase.label);
            System.out.println("Answer: " + findMax(result));
        }
        cnn.print(Paths.get("./resources/mnist_by_class/result").toAbsolutePath().toString());
    }
    
    private static void readTrainFile() throws FileNotFoundException, IOException {
        try (InputStream in1 = new FileInputStream(Paths.get("./resources/mnist_by_class/letters_digits_train").toAbsolutePath().toString());
                InputStream in2 = new FileInputStream(Paths.get("./resources/mnist_by_class/letters_digits_train_label").toAbsolutePath().toString())) {
            System.out.println("Magic Number: " + readInt(in1));
            int images = readInt(in1);
            int rows = readInt(in1);
            int cols = readInt(in1);
            System.out.println("Images: " + images);
            System.out.println("Rows: " + rows);
            System.out.println("Cols: " + cols);
            System.out.println("Magic Number: " + readInt(in2));
            int labels = readInt(in2);
            System.out.println("Labels: " + labels);
            int size = Math.min(images, labels);
            size = 10000; // just for testing, otherwise there would be nearly 700 000 images to compute and train.
            testCases = new TestCase[size];
            for (int i = 0; i < size; i++) {
                testCases[i] = new TestCase(rows, cols);
                in1.read(testCases[i].data);
                testCases[i].label = (byte) in2.read();
            }
        }
        System.out.println("done");
    }
    
    private static int readInt(InputStream in) throws IOException {
        int i = in.read() << 24;
        i = i | (in.read() << 16);
        i = i | (in.read() << 8);
        i = i | in.read();
        return i;
    }
    
    private static double[] toDoubleArray(byte[] arr) {
        double[] out = new double[arr.length];
        for(int i = 0; i < arr.length; i++) {
            out[i] = (double)(arr[i] & 0xff) / 128d - 1d;
        }
        return out;
    }
    
    private static int findMax(double[] arr) {
        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for(int i = 0; i < arr.length; i++) {
            if(arr[i] > max) {
                max = arr[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}