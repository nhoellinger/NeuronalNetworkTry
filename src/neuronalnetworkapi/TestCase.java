package neuronalnetworkapi;

public class TestCase {
    byte[] data;
    byte label;

    public TestCase(int rows, int cols) {
        data = new byte[rows * cols];
    }
}
