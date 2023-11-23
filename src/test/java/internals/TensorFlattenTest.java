package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorFlattenTest {

    @Test
    public void testFlatten() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        Tensor expected = new Tensor(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        assertEquals(expected, tensor.flatten());
    }

    @Test
    public void testDiagFlat() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
        });
        Tensor expected = new Tensor(new int[][]{
                {1, 0, 0, 0, 0, 0},
                {0, 2, 0, 0, 0, 0},
                {0, 0, 3, 0, 0, 0},
                {0, 0, 0, 4, 0, 0},
                {0, 0, 0, 0, 5, 0},
                {0, 0, 0, 0, 0, 6},
        });
        assertEquals(expected, JNDArray.diagFlat(tensor));
    }

}
