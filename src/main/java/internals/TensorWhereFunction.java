package internals;

import java.util.function.Function;

public class TensorWhereFunction {

    static Tensor where(Tensor tensor, Function<Number, Number> function) {
        for (int i = 0; i < tensor.getInternalIndexingTableSize(); i++) {
            tensor.setInInternalArray(i, function.apply(tensor.getFromInternalArray(i)));
        }
        return tensor;
    }

}
