package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import java.util.List;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorEnumerateTest {

    @Test(expected = IllegalArgumentException.class)
    public void testEnumerate_Scalar() {
        Tensor tensor = new Tensor(1);
        List<Tensor> list = tensor.enumerate();
    }


    @Test
    public void testEnumerate_1D() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3});
        List<Tensor> list = tensor.enumerate();
        Tensor expected1 = new Tensor(1);
        Tensor expected2 = new Tensor(2);
        Tensor expected3 = new Tensor(3);
        assertEquals(3, list.size());
        assertEquals(expected1, list.get(0));
        assertEquals(expected2, list.get(1));
        assertEquals(expected3, list.get(2));
    }

    @Test
    public void testEnumerate_2D() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        List<Tensor> list = tensor.enumerate();
        Tensor expected1 = new Tensor(new int[] {1, 2, 3});
        Tensor expected2 = new Tensor(new int[] {4, 5, 6});
        Tensor expected3 = new Tensor(new int[] {7, 8, 9});
        Tensor expected4 = new Tensor(new int[] {10, 11, 12});
        assertEquals(4, list.size());
        assertEquals(expected1, list.get(0));
        assertEquals(expected2, list.get(1));
        assertEquals(expected3, list.get(2));
        assertEquals(expected4, list.get(3));
    }



}
