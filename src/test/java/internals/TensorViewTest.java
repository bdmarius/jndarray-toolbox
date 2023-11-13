package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.*;

@RunWith(MockitoJUnitRunner.class)
public class TensorViewTest {

    @Test
    public void testView_Creation() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        Tensor view = tensor.view();
        assertTrue(view.isView());
        assertEquals(tensor, view.getBase());
        assertEquals(tensor, view);
    }

    @Test
    public void testView_GetAndSet() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        Tensor view = tensor.view();
        assertEquals(5, tensor.get(1, 1));
        assertEquals(5, view.get(1, 1));

        tensor.set(50, 1, 1);
        assertEquals(50, tensor.get(1, 1));
        assertEquals(50, view.get(1, 1));

        view.set(500, 1, 1);
        assertEquals(500, tensor.get(1, 1));
        assertEquals(500, view.get(1, 1));
    }

    @Test
    public void testView_Of_View_GetAndSet() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        Tensor view1 = tensor.view();
        Tensor view2 = view1.view();
        tensor.set(50, 1, 1);

        assertEquals(50, tensor.get(1, 1));
        assertEquals(50, view1.get(1, 1));
        assertEquals(50, view2.get(1, 1));

        view1.set(500, 1, 1);
        assertEquals(500, tensor.get(1, 1));
        assertEquals(500, view1.get(1, 1));
        assertEquals(500, view2.get(1, 1));

        view2.set(5000, 1, 1);
        assertEquals(5000, tensor.get(1, 1));
        assertEquals(5000, view1.get(1, 1));
        assertEquals(5000, view2.get(1, 1));
    }


}
