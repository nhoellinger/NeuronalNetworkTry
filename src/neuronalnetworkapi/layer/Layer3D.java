package neuronalnetworkapi.layer;

public abstract class Layer3D extends Layer {
    int inputDepth;
    int inputLen;
    int kernelSize;
    int stride;
    int padding;
    int outputDepth;
    int outputLen;
    
    public Layer3D(int inputDepth, int inputLen, int kernelSize, int stride, int padding, int outputDepth) {
        super(inputDepth * inputLen * inputLen, ((inputLen - kernelSize + 2 * padding) / stride + 1)
            * ((inputLen - kernelSize + 2 * padding) / stride + 1) * outputDepth);
        this.inputDepth = inputDepth;
        this.inputLen = inputLen;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.outputDepth = outputDepth;
        outputLen = (inputLen - kernelSize + 2 * padding) / stride + 1;
    }
}
