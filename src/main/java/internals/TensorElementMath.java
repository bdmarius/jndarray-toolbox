package internals;

import utils.JNumDataType;
import utils.NumberUtils;
import utils.TypeUtils;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

public class TensorElementMath {

    static Tensor powerOf(Tensor tensor, Number value) {
        return performBiFunctionOperation(tensor, value, NumberUtils.POWER_OF);
    }

    static Tensor log(Tensor tensor) {
        return performFunctionOperation(tensor, NumberUtils.LOG);
    }

    static Tensor exp(Tensor tensor) {
        return performFunctionOperation(tensor, NumberUtils.EXP);
    }

    static Tensor sqrt(Tensor tensor) {
        return performFunctionOperation(tensor, NumberUtils.SQRT);
    }

    static Tensor minus(Tensor tensor) {
        return performFunctionOperation(tensor, NumberUtils.MINUS);
    }

    static Tensor min(Tensor tensor, Number value) {
        return performBiFunctionOperation(tensor, value, NumberUtils.MIN);
    }

    static Tensor max(Tensor tensor, Number value) {
        return performBiFunctionOperation(tensor, value, NumberUtils.MAX);
    }

    static Tensor clip(Tensor tensor, Number minValue, Number maxValue) {
        Tensor firstResult = performBiFunctionOperation(tensor, minValue, NumberUtils.MAX);
        return performBiFunctionOperation(firstResult, maxValue, NumberUtils.MIN);
    }

    private static Tensor performFunctionOperation(Tensor tensor, Map<JNumDataType, Function<Number, Number>> operation) {
        for (int i = 0; i < tensor.getInternalArraySize(); i++) {
            Number valueInTensor = tensor.getFromInternalArray(i);
            tensor.setInInternalArray(i, operation.get(tensor.getDataType()).apply(valueInTensor));
        }
        return tensor;
    }

    private static Tensor performBiFunctionOperation(Tensor tensor, Number value, Map<JNumDataType, Map<JNumDataType, BiFunction<Number, Number, Number>>> operation) {
        JNumDataType valueDataType = TypeUtils.parseDataType(value.getClass());
        for (int i = 0; i < tensor.getInternalArraySize(); i++) {
            Number valueInTensor = tensor.getFromInternalArray(i);
            tensor.setInInternalArray(i, operation.get(tensor.getDataType()).get(valueDataType).apply(valueInTensor, value));
        }
        return tensor;
    }

}
