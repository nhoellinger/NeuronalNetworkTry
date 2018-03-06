package neuronalnetworkapi.layer;

public interface ILayer {
    double[] compute(double[] input);
    void backPropagate(Layer nextLayer);
}
