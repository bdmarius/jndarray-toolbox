package internals;

import utils.ShapeUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class TensorIndices {

    /**
     * Gets all possible indices of a given Tensor
     */
    static List<int[]> indices(Tensor tensor) {
        int maximumNumberOfElements = ShapeUtils.getSizeFromShape(tensor.getShapeList());
        List<int[]> result = new ArrayList<>(maximumNumberOfElements);
        int[] shape = tensor.getShape();
        int[] indexTracker = new int[shape.length];
        boolean firstIteration = true;

        for (int i = 0; i < maximumNumberOfElements; i++) {
            for (int j = indexTracker.length - 1; j >= 0; j--) {
                if (!firstIteration) {
                    indexTracker[j]++;
                } else {
                    firstIteration = false;
                }
                if (indexTracker[j] >= shape[j]) {
                    indexTracker[j] = 0;
                } else {
                    result.add(indexTracker.clone());
                    break;
                }
            }
        }
        return result;
    }

    /**
     * From all possible indices of a given Tensor, returns indices that meet the given condition
     */
    static List<int[]> indices(Tensor tensor, Function<Number, Boolean> function) {
        int maximumNumberOfElements = ShapeUtils.getSizeFromShape(tensor.getShapeList());
        List<int[]> result = new ArrayList<>(maximumNumberOfElements);
        int[] shape = tensor.getShape();
        int[] indexTracker = new int[shape.length];
        boolean firstIteration = true;

        for (int i = 0; i < maximumNumberOfElements; i++) {
            for (int j = indexTracker.length - 1; j >= 0; j--) {
                if (!firstIteration) {
                    indexTracker[j]++;
                } else {
                    firstIteration = false;
                }
                if (indexTracker[j] >= shape[j]) {
                    indexTracker[j] = 0;
                } else {
                    if (function.apply(tensor.get(indexTracker))) {
                        result.add(indexTracker.clone());
                    }
                    break;
                }
            }
        }
        return result;
    }
}
