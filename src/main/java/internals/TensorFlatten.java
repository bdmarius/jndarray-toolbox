package internals;

import java.util.stream.IntStream;

public class TensorFlatten {

    /**
     * Returns a 1-D Tensor with all the values from the original tensor.
     */
    static Tensor flatten(Tensor tensor) {
        int[] currentShape = tensor.getShape();
        int numberOfElements = IntStream.of(currentShape).reduce((a, b) -> a * b).orElse(1);
        return tensor.clone().reshape(new int[] {numberOfElements});
    }

    /**
     * Returns a 2-D tensor where the main diagonal contains all the values from the original tensor
     */
    static Tensor diagFlat(Tensor tensor) {
        int[] currentShape = tensor.getShape();
        int numberOfElements = IntStream.of(currentShape).reduce((a, b) -> a * b).orElse(1);
        Tensor flattenedTensor = tensor.clone().reshape(new int[] {numberOfElements});
        int[] newShape = new int[] {numberOfElements, numberOfElements};
        Tensor result = new Tensor(tensor.getDataType(), newShape);
        for (int i = 0; i < numberOfElements; i++) {
            result.set(flattenedTensor.get(i), i, i);
        }
        return result;
    }

}
