package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorAssignTest {

    @Test
    public void test_Assign_2D() {
        Tensor firstTensor = new Tensor(new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
        });
        Tensor secondTensor = new Tensor(new double[][]{
                {0, 0, 0, 0},
        });
        Tensor expected = new Tensor(new double[][] {
                {1, 2, 3, 4},
                {0, 0, 0, 0},
                {9, 10, 11, 12}
        });
        firstTensor.set(new int[][] {{1, 1}, {0, 3}}, secondTensor);
        assertEquals(expected, firstTensor);

        secondTensor = new Tensor(new double[][]{
                {100},
                {100},
                {100}
        });
        expected = new Tensor(new double[][] {
                {1, 2, 100, 4},
                {0, 0, 100, 0},
                {9, 10, 100, 12}
        });
        firstTensor.set(new int[][] {{0, 2}, {2, 2}}, secondTensor);
        assertEquals(expected, firstTensor);
    }

    @Test(expected = IllegalArgumentException.class)
    public void test_Assign_IllegalLimits() {
        Tensor firstTensor = new Tensor(new double[][]{
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
        });
        Tensor secondTensor = new Tensor(new double[][]{
                {0, 0, 0, 0},
        });
        firstTensor.set(new int[][] {{1, 1}, {0, 4}}, secondTensor);
    }

}
