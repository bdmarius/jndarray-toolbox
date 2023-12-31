package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import utils.JNumDataType;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorGeneratorTest {

    @Test
    public void test_Zeroes() {
        Tensor expected = new Tensor(new int[][]{
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
        });
        assertEquals(expected, JNDArray.zeroes(JNumDataType.INT, new int[]{3, 4}));
    }

    @Test
    public void test_Ones() {
        Tensor expected = new Tensor(new int[][]{
                {1, 1, 1, 1},
                {1, 1, 1, 1},
                {1, 1, 1, 1},
        });
        assertEquals(expected, JNDArray.ones(JNumDataType.INT, new int[]{3, 4}));
    }

    @Test
    public void test_Empty() {
        Tensor expected = new Tensor(new Integer[][]{
                {null, null, null, null},
                {null, null, null, null},
                {null, null, null, null},
        });
        assertEquals(expected, JNDArray.empty(new int[]{3, 4}));
    }

    @Test
    public void test_Identity() {
        Tensor expected = new Tensor(new Integer[][]{
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1},
        });
        assertEquals(expected, JNDArray.identity(JNumDataType.INT, 4));
    }
}
