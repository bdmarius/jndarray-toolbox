package internals;

import utils.NumberUtils;
import utils.TypeUtils;

import java.util.function.Function;

public class TensorLogicFunctions {

    static boolean lower(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.lower(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    static boolean lowerEquals(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.lowerEquals(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    static boolean greater(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.greater(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    static boolean greaterEquals(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.greaterEquals(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    static boolean equals(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.equals(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    static boolean notEquals(Tensor tensor, Number value) {
        return all(tensor, (x ->
                NumberUtils.notEquals(tensor.getDataType(), x, TypeUtils.parseDataType(value.getClass()), value)));
    }

    static Boolean all(Tensor tensor, Function<Number, Boolean> function) {
        return elementWiseAndOperator(tensor, function);
    }

    static Boolean any(Tensor tensor, Function<Number, Boolean> function) {
        return elementWiseOrOperator(tensor, function);
    }

    private static boolean elementWiseOrOperator(Tensor tensor, Function<Number, Boolean> function) {
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            if (function.apply(tensor.getFromInternalArray(i))) {
                return true;
            }
        }
        return false;
    }

    private static boolean elementWiseAndOperator(Tensor tensor, Function<Number, Boolean> function) {
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            if (!function.apply(tensor.getFromInternalArray(i))) {
                return false;
            }
        }
        return true;
    }

}
