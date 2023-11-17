package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorWhereFunctionTest {

    @Test
    public void testWhere() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        Tensor actual = JNDArray.where(tensor, (x -> {
            if (x.doubleValue() >= 5) {
                return x;
            } else {
                return 0;
            }
        }));
        Tensor expected = new Tensor(new int[][]{
                {0, 0, 0},
                {0, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        assertEquals(expected, actual);
    }

}
