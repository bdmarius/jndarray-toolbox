package internals;

import utils.TypeUtils;

import java.util.Arrays;

/**
 * Concatenates 2 tensors on a given axis.
 * Both tensors must be of the same shape, except for the given axis.
 */
public class TensorConcatenate {

    static Tensor concatenate(Tensor firstTensor, Tensor secondTensor, int axis) {
        int[] firstTensorShape = firstTensor.getShape();
        int[] secondTensorShape = secondTensor.getShape();
        if (firstTensorShape.length != secondTensorShape.length) {
            throw new IllegalArgumentException(String.format("Cannot concatenate tensors of shape %s and %s. " +
                            "Shapes must be of equal length",
                    Arrays.toString(firstTensorShape), Arrays.toString(secondTensorShape)));
        }
        if (axis < 0 || axis >= firstTensorShape.length) {
            throw new IllegalArgumentException(String.format("Cannot concatenate tensors of shape %s and %s. " +
                            "Invalid axis %s provided.",
                    Arrays.toString(firstTensorShape), Arrays.toString(secondTensorShape), axis));
        }
        for (int i = 0; i < firstTensorShape.length; i++) {
            if (i != axis && firstTensorShape[i] != secondTensorShape[i]) {
                throw new IllegalArgumentException(String.format("Cannot concatenate tensors of shape %s and %s. " +
                                "Shapes must be equal except for the given axis.",
                        Arrays.toString(firstTensorShape), Arrays.toString(secondTensorShape)));
            }
        }
        int[] newShape = new int[firstTensorShape.length];
        for (int i = 0; i < firstTensorShape.length; i++) {
            newShape[i] = i != axis ? firstTensorShape[i] : firstTensorShape[i] + secondTensorShape[i];
        }
        Tensor result = new Tensor(TypeUtils.getHighestDataType(firstTensor.getDataType(),
                secondTensor.getDataType()), newShape);
        int[] indexTracker = new int[newShape.length];
        int[] indexTrackerCopy = new int[newShape.length];
        boolean firstIteration = true;
        for (int i = 0; i < result.getInternalIndexingTableSize(); i++) {
            for (int j = indexTracker.length - 1; j >= 0; j--) {
                if (!firstIteration) {
                    indexTracker[j]++;
                    indexTrackerCopy[j]++;
                } else {
                    firstIteration = false;
                }
                if (indexTracker[j] >= newShape[j]) {
                    indexTracker[j] = 0;
                    indexTrackerCopy[j] = 0;
                } else {
                    if (indexTracker[axis] >= firstTensorShape[axis]) {
                        indexTrackerCopy[axis] -= firstTensorShape[axis];
                        result.set(secondTensor.get(indexTrackerCopy), indexTracker);
                        indexTrackerCopy[axis] = indexTracker[axis];
                    } else {
                        result.set(firstTensor.get(indexTracker), indexTracker);
                    }
                    break;
                }
            }
        }
        return result;
    }

}
