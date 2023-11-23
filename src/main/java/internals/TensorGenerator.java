package internals;

import utils.JNumDataType;
import utils.TypeUtils;

public class TensorGenerator {

    /**
     * Returns a Tensor of a requested type and shape, filled with 0 values cast to the specific type.
     */
    static Tensor zeroes(JNumDataType dataType, int[] shape) {
        return new Tensor(dataType, shape, TypeUtils::getDefaultValue);
    }

    /**
     * Returns a Tensor of a requested type and shape, filled with 1 values cast to the specific type.
     */
    static Tensor ones(JNumDataType dataType, int[] shape) {
        return new Tensor(dataType, shape, TypeUtils::getOne);
    }

    /**
     * Returns a Tensor of a requested type and shape, filled with null values
     */
    static Tensor empty(int[] shape) {
        return new Tensor(JNumDataType.DOUBLE, shape, TypeUtils::getNull);
    }

}
