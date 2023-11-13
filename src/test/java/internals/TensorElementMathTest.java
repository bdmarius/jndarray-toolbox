package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorElementMathTest {

    @Test
    public void test_Minus_With_View() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor tensor2 = JNum.broadcast(tensor, new int[]{4, 5});
        Tensor expectedResult = new Tensor(new int[][]{
                new int[]{-1, -2, -3, -4, -5},
                new int[]{-1, -2, -3, -4, -5},
                new int[]{-1, -2, -3, -4, -5},
                new int[]{-1, -2, -3, -4, -5}
        });
        assertEquals(expectedResult, tensor2.minus());
    }

    @Test
    public void test_Clip_With_View() {
        Tensor tensor = new Tensor(new int[]{1, 2, 3, 4, 5});
        Tensor tensor2 = JNum.broadcast(tensor, new int[]{4, 5});
        Tensor expectedResult = new Tensor(new int[][]{
                new int[]{2, 2, 3, 4, 4},
                new int[]{2, 2, 3, 4, 4},
                new int[]{2, 2, 3, 4, 4},
                new int[]{2, 2, 3, 4, 4},
        });
        assertEquals(expectedResult, tensor2.clip(2, 4));
    }

    @Test
    public void test_PowerOf() {
        Tensor tensor = new Tensor(new int[][] {
                new int[] {1, 2, 3},
                new int[] {4, 5, 6}
        });
        Tensor actual = JNum.powerOf(tensor, 2);
        Tensor expected = new Tensor(new double[][] {
                new double[] {1.0, 4.0, 9.0},
                new double[] {16.0, 25.0, 36.0}
        });

        assertEquals(expected, actual);
    }

    @Test
    public void test_log() {
        Tensor tensor = new Tensor(new int[][] {
                new int[] {1, 2, 3},
                new int[] {4, 5, 6}
        });
        Tensor actual = JNum.log(tensor);
        Tensor expected = new Tensor(new double[][] {
                new double[] {0.0, 0.6931471805599453, 1.0986122886681098},
                new double[] {1.3862943611198906, 1.6094379124341003, 1.791759469228055}
        });

        assertEquals(expected, actual);
    }

    @Test
    public void test_exp() {
        Tensor tensor = new Tensor(new int[][] {
                new int[] {1, 2, 3},
                new int[] {4, 5, 6}
        });
        Tensor actual = JNum.exp(tensor);
        Tensor expected = new Tensor(new double[][] {
                new double[] {2.718281828459045, 7.3890560989306495, 20.085536923187664},
                new double[] {54.59815003314423, 148.41315910257657, 403.428793492735}
        });

        assertEquals(expected, actual);
    }

    @Test
    public void test_sqrt() {
        Tensor tensor = new Tensor(new int[][] {
                new int[] {1, 2, 3},
                new int[] {4, 5, 6}
        });
        Tensor actual = JNum.sqrt(tensor);
        Tensor expected = new Tensor(new double[][] {
                new double[] {1.0, 1.4142135623730951, 1.7320508075688772},
                new double[] {2.0, 2.23606797749979, 2.449489742783178}
        });

        assertEquals(expected, actual);
    }

    @Test
    public void test_Minus() {
        Tensor tensor = new Tensor(new int[][] {
                new int[] {1, 2, 3},
                new int[] {4, 5, 6}
        });
        Tensor actual = JNum.minus(tensor);
        Tensor expected = new Tensor(new int[][] {
                new int[] {-1, -2, -3},
                new int[] {-4, -5, -6}
        });

        assertEquals(expected, actual);
    }

    @Test
    public void test_Min() {
        Tensor tensor = new Tensor(new int[][] {
                new int[] {1, 2, 3},
                new int[] {4, 5, 6}
        });
        Tensor actual = JNum.min(tensor, 3);
        Tensor expected = new Tensor(new int[][] {
                new int[] {1, 2, 3},
                new int[] {3, 3, 3}
        });

        assertEquals(expected, actual);
    }

    @Test
    public void test_Max() {
        Tensor tensor = new Tensor(new int[][] {
                new int[] {1, 2, 3},
                new int[] {4, 5, 6}
        });
        Tensor actual = JNum.max(tensor, 3);
        Tensor expected = new Tensor(new int[][] {
                new int[] {3, 3, 3},
                new int[] {4, 5, 6}
        });

        assertEquals(expected, actual);
    }

    @Test
    public void test_Clip() {
        Tensor tensor = new Tensor(new int[][] {
                new int[] {1, 2, 3},
                new int[] {4, 5, 6}
        });
        Tensor actual = JNum.clip(tensor, 2, 4);
        Tensor expected = new Tensor(new int[][] {
                new int[] {2, 2, 3},
                new int[] {4, 4, 4}
        });

        assertEquals(expected, actual);
    }
}

