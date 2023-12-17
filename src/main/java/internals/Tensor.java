package internals;

import utils.JNumDataType;
import utils.TypeUtils;

import java.lang.reflect.Array;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * The Tensor class is the central part of the JNum library. A Tensor is an N-Dimensional (N>=0) array and the Tensor
 * class holds this data as well as other metadata useful in different operations. Tensors are homogeneous, meaning
 * all items in a tensor can and will be all of one type.
 * All tensors have a shape, a list of integers, where shape[i] = the size of the tensor in dimension i.
 * Tensor values are stored in a contiguous array, regardless of their shape, and tensors are equipped with strides
 * which help us determine which dimension every item belongs to.
 * The strides will be the number of elements we need to jump to move between elements of the same dimension.
 * For example, for a 2-D array of 3 columns (number of rows irrelevant), the strides will be [3, 1]
 * We need to jump 3 elements in the internal array to navigate between row 0 and row 1 in the same column.
 * We need to jump 1 element in the internal array to navigate between column 0 and column 1 in the same row.
 * Strides are therefore useful when we need to access one specific element in the internal array. In a [4, 2] array,
 * the strides will be (4, 1), so for an element on line 2, column 0, we will compute the internal index like
 * 4*2 + 1*0 = 8.
 * Each Tensor has a dataType which tells us which child of the Java Number class is this Tensor instances supposed to
 * hold.
 * A Tensor instance can also be a view of another Tensor (called a base). This is important because many operations
 * that we apply to a Tensor do not change the actual internal array, but only change its metadata, which makes a Tensor
 * "look" like another view of a base Tensor. This is done for speed and memory optimisation purposes.
 * The Tensors get initialised with an indexing table (0, 1, 2, ..., n-1) which can be re-written in operations
 * such as Transpose, Reshape or Broadcast. Therefore, after such an operation, the n-th element of an
 * internalIndexingTable of a view Tensor can point to the m-th element of its base Tensor.
 */
public final class Tensor {

    private List<Integer> shape = new ArrayList<>();
    private List<Integer> strides = new ArrayList<>();
    private Number[] internalArray;
    private JNumDataType dataType;
    private int numberOfElements = 0;
    private boolean isView;
    private Tensor base;
    private int[] internalIndexingTable;

    public Tensor(Object data) {
        buildShape(data);
        buildStrides();
        int numberOfElements = shape.stream().reduce(1, (i, j) -> i * j);
        internalArray = new Number[numberOfElements];
        buildInternalArray(data);
        buildInternalIndexingTable();
        Number firstValue = getValue(0);
        if (firstValue != null) {
            dataType = TypeUtils.parseDataType(firstValue.getClass());
        } else {
            dataType = JNumDataType.DOUBLE;
        }
        isView = false;
        base = null;
    }

    public Tensor(JNumDataType dataType, int[] shape) {
        this.shape = Arrays.stream(shape).boxed().collect(Collectors.toList());
        this.dataType = dataType;
        buildStrides();
        int numberOfElements = this.shape.stream().reduce(1, (i, j) -> i * j);
        buildDefaultInternalArray(numberOfElements, TypeUtils::getDefaultValue);
        buildInternalIndexingTable();
        isView = false;
        base = null;
    }

    public Number get(int... index) {
        return getValue(index);
    }

    public void set(Number newValue, int... index) {
        setValue(newValue, index);
    }

    public void set(int[][] limits, Tensor secondTensor) {
        JNDArray.set(limits, this, secondTensor);
    }

    public Tensor view() {
        Tensor result = this.clone();
        result.numberOfElements = 0;
        result.internalArray = null;
        result.internalIndexingTable = this.internalIndexingTable.clone();
        result.isView = true;
        result.base = this;
        return result;
    }

    public Tensor transposed() {
        return JNDArray.transpose(this);
    }

    public Tensor reshape(int[] newShape) {
        return JNDArray.reshape(this, newShape);
    }

    public Tensor broadcast(int[] newShape) {
        return JNDArray.broadcast(this, newShape);
    }

    public Tensor add(Tensor tensor) {
        return JNDArray.add(this, tensor);
    }

    public Tensor subtract(Tensor tensor) {
        return JNDArray.subtract(this, tensor);
    }

    public Tensor multiply(Tensor tensor) {
        return JNDArray.multiply(this, tensor);
    }

    public Tensor divide(Tensor tensor) {
        return JNDArray.divide(this, tensor);
    }

    public Tensor add(Number value) {
        return JNDArray.add(this, value);
    }

    public Tensor subtract(Number value) {
        return JNDArray.subtract(this, value);
    }

    public Tensor multiply(Number value) {
        return JNDArray.multiply(this, value);
    }

    public Tensor divide(Number value) {
        return JNDArray.divide(this, value);
    }

    public Tensor powerOf(Number value) {
        return JNDArray.powerOf(this, value);
    }

    public Tensor log() {
        return JNDArray.log(this);
    }

    public Tensor exp() {
        return JNDArray.exp(this);
    }

    public Tensor sqrt() {
        return JNDArray.sqrt(this);
    }

    public Tensor minus() {
        return JNDArray.minus(this);
    }

    public Tensor min(Number value) {
        return JNDArray.min(this, value);
    }

    public Tensor max(Number value) {
        return JNDArray.max(this, value);
    }

    public Tensor clip(Number firstValue, Number secondValue) {
        return JNDArray.clip(this, firstValue, secondValue);
    }

    public Tensor slice(int[][] limits) {
        return JNDArray.slice(this, limits);
    }

    public Tensor dot(Tensor secondTensor) {
        return JNDArray.dot(this, secondTensor);
    }

    public Tensor min() {
        return JNDArray.min(this);
    }

    public Tensor min(boolean keepDimensions) {
        return JNDArray.min(this, keepDimensions);
    }

    public Tensor min(int[] axis) {
        return JNDArray.min(this, axis);
    }

    public Tensor min(int[] axis, boolean keepDimensions) {
        return JNDArray.min(this, axis, keepDimensions);
    }

    public Tensor max() {
        return JNDArray.max(this);
    }

    public Tensor max(boolean keepDimensions) {
        return JNDArray.max(this, keepDimensions);
    }

    public Tensor max(int[] axis) {
        return JNDArray.max(this, axis);
    }

    public Tensor max(int[] axis, boolean keepDimensions) {
        return JNDArray.max(this, axis, keepDimensions);
    }

    public Tensor argMin() {
        return JNDArray.argMin(this);
    }

    public Tensor argMin(boolean keepDimensions) {
        return JNDArray.argMin(this, keepDimensions);
    }

    public Tensor argMin(int axis) {
        return JNDArray.argMin(this, axis);
    }

    public Tensor argMin(int axis, boolean keepDimensions) {
        return JNDArray.argMin(this, axis, keepDimensions);
    }

    public Tensor argMax() {
        return JNDArray.argMax(this);
    }

    public Tensor argMax(boolean keepDimensions) {
        return JNDArray.argMax(this, keepDimensions);
    }

    public Tensor argMax(int axis) {
        return JNDArray.argMax(this, axis);
    }

    public Tensor argMax(int axis, boolean keepDimensions) {
        return JNDArray.argMax(this, axis, keepDimensions);
    }

    public Tensor mean() {
        return JNDArray.mean(this);
    }

    public Tensor mean(boolean keepDimensions) {
        return JNDArray.mean(this, keepDimensions);
    }

    public Tensor mean(int[] axis) {
        return JNDArray.mean(this, axis);
    }

    public Tensor mean(int[] axis, boolean keepDimensions) {
        return JNDArray.mean(this, axis, keepDimensions);
    }

    public Tensor median() {
        return JNDArray.median(this);
    }

    public Tensor median(boolean keepDimensions) {
        return JNDArray.median(this, keepDimensions);
    }

    public Tensor median(int[] axis) {
        return JNDArray.median(this, axis);
    }

    public Tensor median(int[] axis, boolean keepDimensions) {
        return JNDArray.median(this, axis, keepDimensions);
    }

    public Tensor mode() {
        return JNDArray.mode(this);
    }

    public Tensor mode(boolean keepDimensions) {
        return JNDArray.mode(this, keepDimensions);
    }

    public Tensor mode(int[] axis) {
        return JNDArray.mode(this, axis);
    }

    public Tensor mode(int[] axis, boolean keepDimensions) {
        return JNDArray.mode(this, axis, keepDimensions);
    }

    public Tensor std() {
        return JNDArray.std(this);
    }

    public Tensor std(boolean keepDimensions) {
        return JNDArray.std(this, keepDimensions);
    }

    public Tensor std(int[] axis) {
        return JNDArray.std(this, axis);
    }

    public Tensor std(int[] axis, boolean keepDimensions) {
        return JNDArray.std(this, axis, keepDimensions);
    }

    public Tensor sum() {
        return JNDArray.sum(this);
    }

    public Tensor sum(boolean keepDimensions) {
        return JNDArray.sum(this, keepDimensions);
    }

    public Tensor sum(int[] axis) {
        return JNDArray.sum(this, axis);
    }

    public Tensor sum(int[] axis, boolean keepDimensions) {
        return JNDArray.sum(this, axis, keepDimensions);
    }

    public Tensor prod() {
        return JNDArray.prod(this);
    }

    public Tensor prod(boolean keepDimensions) {
        return JNDArray.prod(this, keepDimensions);
    }

    public Tensor prod(int[] axis) {
        return JNDArray.prod(this, axis);
    }

    public Tensor prod(int[] axis, boolean keepDimensions) {
        return JNDArray.prod(this, axis, keepDimensions);
    }

    public boolean lower(Number value) {
        return JNDArray.lower(this, value);
    }

    public boolean lowerEquals(Number value) {
        return JNDArray.lowerEquals(this, value);
    }

    public boolean greater(Number value) {
        return JNDArray.greater(this, value);
    }

    public boolean greaterEquals(Number value) {
        return JNDArray.greaterEquals(this, value);
    }

    public boolean equals(Number value) {
        return JNDArray.equals(this, value);
    }

    public boolean notEquals(Number value) {
        return JNDArray.notEquals(this, value);
    }

    public boolean all(Function<Number, Boolean> function) {
        return JNDArray.all(this, function);
    }

    public boolean any(Function<Number, Boolean> function) {
        return JNDArray.any(this, function);
    }

    public Tensor where(Function<Number, Number> function) {
        return JNDArray.where(this, function);
    }

    public List<int[]> indices() {
        return JNDArray.indices(this);
    }

    public List<int[]> indices(Function<Number, Boolean> function) {
        return JNDArray.indices(this, function);
    }

    public Tensor flatten() {
        return JNDArray.flatten(this);
    }

    public List<Tensor> enumerate() {
        return JNDArray.enumerate(this);
    }

    public List<Number> getValues() {
        return JNDArray.getValues(this);
    }

    public Tensor concatenate(Tensor secondTensor, int axis) {
        return JNDArray.concatenate(this, secondTensor, axis);
    }

    /**
     * Returns true if all elements are equal with a small delta
     */
    public boolean equals(Object o, double delta) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Tensor tensor = (Tensor) o;
        if (!Arrays.equals(getShape(), tensor.getShape())) {
            return false;
        }

        for (int i = 0; i < internalIndexingTable.length; i++) {
            if (Math.abs(getFromInternalArray(i).doubleValue() - tensor.getFromInternalArray(i).doubleValue()) > delta) {
                return false;
            }
        }
        return true;
    }

    /**
     * In order to know where to put brackets, we look at the shape array in reverse.
     * Before every element of index divisible with "shape[n-1]" we append "]\n["
     * Before every element of index divisible with "shape[n-1] * shape[n-2]" we append "]]\n[[" and so on, except for shape[0]
     */
    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("Tensor{" + "shape=").append(Arrays.toString(shape.toArray())).append('}').append(System.lineSeparator());
        StringBuilder ending = new StringBuilder();
        for (int i = 0; i < shape.size(); i++) {
            result.append("[");
            ending.append("]");
        }
        List<Integer> bracketsPositions = new ArrayList<>();
        if (shape.size() > 1) {
            bracketsPositions.add(shape.get(shape.size() - 1));
        }
        for (int i = shape.size() - 2; i >= 1; i--) {
            bracketsPositions.add(shape.get(i) * shape.get(i + 1));
        }
        for (int i = 0; i < internalIndexingTable.length; i++) {
            int countBrackets = 0;
            for (Integer bracketsPosition : bracketsPositions) {
                if (i > 0 && i % bracketsPosition == 0) {
                    countBrackets++;
                }
            }
            if (countBrackets > 0) {
                result.append("]".repeat(countBrackets));
                result.append(System.lineSeparator());
                result.append("[".repeat(countBrackets));
            }
            result.append(" ").append(getFromInternalArray(i)).append(" ");
        }
        return result.append(ending.toString()).toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Tensor tensor = (Tensor) o;
        if (!Arrays.equals(getShape(), tensor.getShape())) {
            return false;
        }

        for (int i = 0; i < internalIndexingTable.length; i++) {
            Number valueFromThisTensor = getFromInternalArray(i);
            Number valueFromOtherTensor = tensor.getFromInternalArray(i);
            if ((valueFromThisTensor == null && valueFromOtherTensor != null) ||
                    (valueFromThisTensor != null && valueFromOtherTensor == null)) {
                return false;
            }
            if (!Objects.equals(valueFromThisTensor, valueFromOtherTensor)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public Tensor clone() {
        Tensor tensor = new Tensor();
        tensor.shape = new ArrayList<>(this.shape.size());
        tensor.shape.addAll(this.shape);
        tensor.strides = new ArrayList<>(this.strides.size());
        tensor.strides.addAll(this.strides);
        if (this.internalArray != null) {
            tensor.internalArray = Arrays.copyOf(this.internalArray, this.internalArray.length);
        }
        if (this.internalIndexingTable != null) {
            tensor.internalIndexingTable = Arrays.copyOf(this.internalIndexingTable, this.internalIndexingTable.length);
        }
        tensor.isView = isView;
        tensor.dataType = this.dataType;
        tensor.numberOfElements = this.numberOfElements;
        return tensor;
    }

    public int[] getShape() {
        return shape.stream().mapToInt(i -> i).toArray();
    }

    public int[] getStrides() {
        return strides.stream().mapToInt(i -> i).toArray();
    }

    public boolean isView() {
        return isView;
    }

    public Tensor getBase() {
        return this.base;
    }

    public JNumDataType getDataType() {
        return dataType;
    }

    Tensor(JNumDataType dataType, int[] shape, Function<JNumDataType, Number> defaultValueGenerator) {
        this.shape = Arrays.stream(shape).boxed().collect(Collectors.toList());
        this.dataType = dataType;
        buildStrides();
        int numberOfElements = this.shape.stream().reduce(1, (i, j) -> i * j);
        buildDefaultInternalArray(numberOfElements, defaultValueGenerator);
        buildInternalIndexingTable();
        isView = false;
        base = null;
    }

    List<Integer> getShapeList() {
        return this.shape;
    }

    List<Integer> getStridesList() {
        return this.strides;
    }

    void setStrides(List<Integer> strides) {
        this.strides = strides;
    }

    void setShape(List<Integer> shape) {
        this.shape = shape;
    }

    int[] getInternalIndexingTable() {
        return internalIndexingTable;
    }

    int getInternalIndexingTableSize() {
        return internalIndexingTable.length;
    }

    Number getFromInternalArray(int index) {
        if (isView) {
            if (base.isView) {
                return base.getFromInternalArray(index);
            }
            return base.internalArray[internalIndexingTable[index]];
        }
        return internalArray[internalIndexingTable[index]];
    }

    int getTranslatedIndex(int index) {
        if (isView) {
            if (base.isView) {
                return base.getTranslatedIndex(index);
            }
            return base.internalIndexingTable[index];
        }
        return internalIndexingTable[index];
    }

    void setInInternalArray(int index, Number value) {
        if (isView) {
            if (base.isView) {
                base.setInInternalArray(index, value);
            } else {
                if (!base.dataType.equals(TypeUtils.parseDataType(value.getClass()))) {
                    base.dataType = TypeUtils.getHighestDataType(base.dataType, TypeUtils.parseDataType(value.getClass()));
                }
                base.internalArray[index] = value;
            }
        } else {
            if (!this.dataType.equals(TypeUtils.parseDataType(value.getClass()))) {
                this.dataType = TypeUtils.getHighestDataType(this.dataType, TypeUtils.parseDataType(value.getClass()));
            }
            internalArray[index] = value;
        }
    }

    int getInternalArraySize() {
        if (isView) {
            if (base.isView) {
                return base.getInternalArraySize();
            }
            return base.internalArray.length;
        }
        return internalArray.length;
    }

    void setInternalIndexingTable(int[] internalIndexingTable) {
        this.internalIndexingTable = internalIndexingTable;
    }

    void setBase(Tensor base) {
        this.base = base;
    }

    private int computeIndex(List<Integer> shape, List<Integer> strides, int... index) {
        if (index == null || index.length == 0) {
            throw new IllegalArgumentException("Cannot return value for null indices");
        }
        int result = 0;
        if (shape.size() > 0) {
            for (int i = 0; i < index.length; i++) {
                if (index[i] < 0 || index[i] >= shape.get(i)) {
                    throw new IllegalArgumentException(String.format("Cannot find index %s in tensor of shape %s", Arrays.toString(index), Arrays.toString(shape.toArray())));
                }
                result += index[i] * strides.get(i);
            }
        }
        return internalIndexingTable[result];
    }

    /**
     * Because we don't know in advance how many dimensions the input data will have, we need to go recursively
     * until the element is not an array anymore.
     * We are also checking whether the array is homogeneous - all arrays in a dimension need to have the same length
     */
    private void buildShape(Object data) {
        if (data != null && data.getClass().isArray()) {
            int length = Array.getLength(data);
            shape.add(length);
            Object nextDimension = Array.get(data, 0);
            if (nextDimension != null && nextDimension.getClass().isArray() && length > 1) {
                int elementLength = Array.getLength(Array.get(data, 0));
                for (int i = 1; i < length; i++) {
                    Object element = Array.get(data, i);
                    if (Array.getLength(element) != elementLength) {
                        throw new IllegalArgumentException("All arrays passed to JNum Tensors need to be homogeneous shape");
                    }
                }
            }
            buildShape(nextDimension);
        }
    }

    private void buildStrides() {
        int currentStride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides.add(currentStride);
            currentStride *= this.shape.get(i);
        }
        Collections.reverse(strides);
    }

    private void buildInternalArray(Object object) {
        if (object != null && object.getClass().isArray()) {
            int length = Array.getLength(object);
            for (int i = 0; i < length; i++) {
                Object arrayElement = Array.get(object, i);
                buildInternalArray(arrayElement);
            }
        } else {
            internalArray[numberOfElements] = (Number) object;
            numberOfElements++;
        }
    }

    private void buildDefaultInternalArray(int numberOfElements, Function<JNumDataType, Number> valueGenerator) {
        internalArray = new Number[numberOfElements];
        for (int i = 0; i < numberOfElements; i++) {
            internalArray[i] = valueGenerator.apply(dataType);
        }
    }

    private void buildInternalIndexingTable() {
        internalIndexingTable = new int[internalArray.length];
        for (int i = 0; i < internalArray.length; i++) {
            internalIndexingTable[i] = i;
        }
    }

    private Number getValue(int... index) {
        if (isView) {
            if (base.isView) {
                return base.getValue(index);
            }
            return base.internalArray[computeIndex(shape, strides, index)];
        }
        return internalArray[computeIndex(shape, strides, index)];
    }

    private void setValue(Number value, int... index) {
        if (isView) {
            base.setValue(value, index);
        } else {
        if (!this.dataType.equals(TypeUtils.parseDataType(value.getClass()))) {
            this.dataType = TypeUtils.getHighestDataType(this.dataType, TypeUtils.parseDataType(value.getClass()));
        }
        internalArray[computeIndex(shape, strides, index)] = value;
        }
    }

    private Tensor() {
        // Empty constructor needed for clone()
    }

}
