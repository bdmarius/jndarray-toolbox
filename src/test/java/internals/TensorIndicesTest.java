package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorIndicesTest {

    @Test
    public void testIndices() {
        Tensor tensor = new Tensor(new int[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12}
        });
        List<int[]> actual1 = tensor.indices();
        List<int[]> expected1 = Arrays.asList(
                new int[] {0, 0},
                new int[] {0, 1},
                new int[] {0, 2},
                new int[] {1, 0},
                new int[] {1, 1},
                new int[] {1, 2},
                new int[] {2, 0},
                new int[] {2, 1},
                new int[] {2, 2},
                new int[] {3, 0},
                new int[] {3, 1},
                new int[] {3, 2}
        );
        assertEquals(expected1.size(), actual1.size());
        for (int i = 0; i < expected1.size(); i++) {
            assertArrayEquals(expected1.get(i), actual1.get(i));
        }

        List<int[]> actual2 = tensor.indices(x -> x.intValue() >= 5);
        List<int[]> expected2 = Arrays.asList(
                new int[] {1, 1},
                new int[] {1, 2},
                new int[] {2, 0},
                new int[] {2, 1},
                new int[] {2, 2},
                new int[] {3, 0},
                new int[] {3, 1},
                new int[] {3, 2}
        );
        assertEquals(expected2.size(), actual2.size());
        for (int i = 0; i < expected2.size(); i++) {
            assertArrayEquals(expected2.get(i), actual2.get(i));
        }
    }

    @Test
    public void testIndices_Views() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor view = JNDArray.broadcast(tensor, new int[]{4, 5});
        List<int[]> actual1 = view.indices();
        List<int[]> expected1 = Arrays.asList(
                new int[] {0, 0},
                new int[] {0, 1},
                new int[] {0, 2},
                new int[] {0, 3},
                new int[] {0, 4},
                new int[] {1, 0},
                new int[] {1, 1},
                new int[] {1, 2},
                new int[] {1, 3},
                new int[] {1, 4},
                new int[] {2, 0},
                new int[] {2, 1},
                new int[] {2, 2},
                new int[] {2, 3},
                new int[] {2, 4},
                new int[] {3, 0},
                new int[] {3, 1},
                new int[] {3, 2},
                new int[] {3, 3},
                new int[] {3, 4}
        );
        assertEquals(expected1.size(), actual1.size());
        for (int i = 0; i < expected1.size(); i++) {
            assertArrayEquals(expected1.get(i), actual1.get(i));
        }

        List<int[]> actual2 = view.indices(x -> x.intValue() >= 3);
        List<int[]> expected2 = Arrays.asList(
                new int[] {0, 2},
                new int[] {0, 3},
                new int[] {0, 4},
                new int[] {1, 2},
                new int[] {1, 3},
                new int[] {1, 4},
                new int[] {2, 2},
                new int[] {2, 3},
                new int[] {2, 4},
                new int[] {3, 2},
                new int[] {3, 3},
                new int[] {3, 4}
        );
        assertEquals(expected2.size(), actual2.size());
        for (int i = 0; i < expected2.size(); i++) {
            assertArrayEquals(expected2.get(i), actual2.get(i));
        }
    }

}
