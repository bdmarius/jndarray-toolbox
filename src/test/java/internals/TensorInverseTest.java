package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorInverseTest {

    @Test
    public void test_Inverse() {
        Tensor tensor = new Tensor(new double[][]{
                {1, 2, 3},
                {3, 2, 1},
                {2, 1, 3},
        });

        Tensor expected = new Tensor(new double[][]{
                {-0.4166666666666667, 0.25, 0.3333333333333333},
                {0.5833333333333334, 0.25, -0.6666666666666666},
                {0.08333333333333333, -0.25, 0.3333333333333333},
        });

        Tensor actual = tensor.inverse();
        assertEquals(expected, actual);
    }

    @Test (expected = IllegalArgumentException.class)
    public void test_Inverse_WrongTensor() {
        Tensor tensor = new Tensor(new double[] {1, 2, 3});
        tensor.inverse();
    }

}
