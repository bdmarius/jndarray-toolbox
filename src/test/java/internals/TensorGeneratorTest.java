package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

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
        assertEquals(expected, JNum.zeroes(JNumDataType.INT, new int[]{3, 4}));
    }

    @Test
    public void test_Ones() {
        Tensor expected = new Tensor(new int[][]{
                {1, 1, 1, 1},
                {1, 1, 1, 1},
                {1, 1, 1, 1},
        });
        assertEquals(expected, JNum.ones(JNumDataType.INT, new int[]{3, 4}));
    }
}
