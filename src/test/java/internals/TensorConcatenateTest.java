package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorConcatenateTest {

    @Test
    public void test_Concatenate() {
        Tensor firstTensor = new Tensor(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor secondTensor = new Tensor(new double[][] {
                {7, 8, 9}
        });
        Tensor expected = new Tensor(new double[][] {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        Tensor actual = firstTensor.concatenate(secondTensor, 0);
        assertEquals(expected, actual);

        secondTensor = new Tensor(new double[][] {
                {7},
                {8}
        });
        expected = new Tensor(new double[][] {
                {1, 2, 3, 7},
                {4, 5, 6, 8}
        });
        actual = firstTensor.concatenate(secondTensor, 1);
        assertEquals(expected, actual);
    }

    @Test(expected = IllegalArgumentException.class)
    public void test_Concatenate_WrongAxis() {
        Tensor firstTensor = new Tensor(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor secondTensor = new Tensor(new double[][] {
                {7, 8, 9}
        });
        firstTensor.concatenate(secondTensor, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void test_Concatenate_WrongShapes() {
        Tensor firstTensor = new Tensor(new double[][] {
                {1, 2, 3},
                {4, 5, 6}
        });
        Tensor secondTensor = new Tensor(new double[][] {
                {7, 8, 9, 10, 11, 12}
        });
        firstTensor.concatenate(secondTensor, 0);
    }
}
