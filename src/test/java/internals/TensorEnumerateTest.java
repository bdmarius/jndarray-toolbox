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

    @Test
    public void testGetValues_2D() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        List<Number> list = tensor.getValues();
        assertEquals(12, list.size());
        assertEquals(1, list.get(0));
        assertEquals(2, list.get(1));
        assertEquals(3, list.get(2));
        assertEquals(4, list.get(3));
        assertEquals(5, list.get(4));
        assertEquals(6, list.get(5));
        assertEquals(7, list.get(6));
        assertEquals(8, list.get(7));
        assertEquals(9, list.get(8));
        assertEquals(10, list.get(9));
        assertEquals(11, list.get(10));
        assertEquals(12, list.get(11));
    }

    @Test
    public void testGetValues_FromView() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor view = JNDArray.broadcast(tensor, new int[]{4, 5});
        List<Number> list = view.getValues();
        assertEquals(20, list.size());
        assertEquals(1, list.get(0));
        assertEquals(2, list.get(1));
        assertEquals(3, list.get(2));
        assertEquals(4, list.get(3));
        assertEquals(5, list.get(4));
        assertEquals(1, list.get(5));
        assertEquals(2, list.get(6));
        assertEquals(3, list.get(7));
        assertEquals(4, list.get(8));
        assertEquals(5, list.get(9));
        assertEquals(1, list.get(10));
        assertEquals(2, list.get(11));
        assertEquals(3, list.get(12));
        assertEquals(4, list.get(13));
        assertEquals(5, list.get(14));
        assertEquals(1, list.get(15));
        assertEquals(2, list.get(16));
        assertEquals(3, list.get(17));
        assertEquals(4, list.get(18));
        assertEquals(5, list.get(19));
    }



}
