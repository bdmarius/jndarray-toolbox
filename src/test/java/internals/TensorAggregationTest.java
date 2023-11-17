package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorAggregationTest {

    @Test
    public void test_Sum_Scalar() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(5);
        assertEquals(expected, tensor.sum());
    }

    @Test
    public void test_Sum_1D() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(6);
        assertEquals(expected, tensor.sum());
    }

    @Test
    public void test_Sum_2D() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(36);
        assertEquals(expected, tensor.sum());
    }

    @Test
    public void test_Sum_3D() {
        Tensor tensor = new Tensor(new int[][][]{
                {
                        {19, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12}
                },
                {
                        {13, 14, 15, 16},
                        {17, 18, 1, 20},
                        {21, 22, 23, 24},
                }
        });
        Tensor expected = new Tensor(300);
        assertEquals(expected, tensor.sum());
    }

    @Test
    public void test_Sum_2D_View() {
        Tensor tensor = new Tensor(new int[]{3, 2, 1, 4, 5});
        Tensor view = JNDArray.broadcast(tensor, new int[]{4, 5});
        Tensor expected = new Tensor(60);
        assertEquals(expected, view.sum());
    }
    
    @Test
    public void test_Sum_1D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(new int[] {6});
        assertEquals(expected, tensor.sum(true));
    }

    @Test
    public void test_Sum_2D_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(new double[][] {{36}});
        assertEquals(expected, tensor.sum(true));
    }

    @Test
    public void test_Sum_3D_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][][]{
                {
                        {19, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12}
                },
                {
                        {13, 14, 15, 16},
                        {17, 18, 1, 20},
                        {21, 22, 23, 24},
                }
        });
        Tensor expected = new Tensor(new double[][][] {{{300}}});
        assertEquals(expected, tensor.sum(true));
    }

    @Test
    public void test_Sum_1D_WithAxis() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(40);
        assertEquals(expected, tensor.sum(new int[] {0}));
    }

    @Test
    public void test_Sum_2D_WithAxis() {
        Tensor tensor = new Tensor(new double[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new double[]
                {26.0, 39.0, 16.0, 34.0}
        );
        assertEquals(expected, tensor.sum(new int[] {0}));

        expected = new Tensor(new double[]
                {40.0, 11.0, 64.0}
        );
        assertEquals(expected, tensor.sum(new int[] {1}));

        expected = new Tensor(115.0);
        assertEquals(expected, tensor.sum(new int[] {0, 1}));
    }

    @Test
    public void test_Sum_3D_WithAxis() {
        Tensor tensor = new Tensor(new double[][][]{
                {
                        {4, 6, 7, 8},
                        {1, 2, 3, 4},
                        {9, 10, 11, 12}
                },
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        });
        Tensor expected = new Tensor(new double[][] {
                {9.0, 24.0, 4.0, 28.0},
                {1, 1, -1, 20},
                {30, 32, 34, 10}
        });
        assertEquals(expected, tensor.sum(new int[] {0}));

        expected = new Tensor(new double[][] {
                {14, 18, 21, 24},
                {26, 39, 16, 34},
        });
        assertEquals(expected, tensor.sum(new int[] {1}));

        expected = new Tensor(new double[][] {
                {25, 10, 42},
                {40, 11, 64},
        });
        assertEquals(expected, tensor.sum(new int[] {2}));

        expected = new Tensor(new double[] {40, 57, 37, 58});
        assertEquals(expected, tensor.sum(new int[] {0, 1}));

        expected = new Tensor(new double[] {77, 115});
        assertEquals(expected, tensor.sum(new int[] {1, 2}));

        expected = new Tensor(192.0);
        assertEquals(expected, tensor.sum(new int[] {0, 1, 2}));
    }

    @Test
    public void test_Sum_1D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(new double[] {40});
        assertEquals(expected, tensor.sum(new int[] {0}, true));
    }

    @Test
    public void test_Sum_2D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new double[][]
                {{26.0, 39.0, 16.0, 34.0}}
        );
        assertEquals(expected, tensor.sum(new int[] {0}, true));

        expected = new Tensor(new double[][]
                {{40.0}, {11.0}, {64.0}}
        );
        assertEquals(expected, tensor.sum(new int[] {1}, true));

        expected = new Tensor(new double[][]{{115.0}});
        assertEquals(expected, tensor.sum(new int[] {0, 1}, true));
    }

    @Test
    public void test_Sum_3D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][][]{
                {
                        {4, 6, 7, 8},
                        {1, 2, 3, 4},
                        {9, 10, 11, 12}
                },
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        });
        Tensor expected = new Tensor(new double[][][] {{
                {9.0, 24.0, 4.0, 28.0},
                {1, 1, -1, 20},
                {30, 32, 34, 10}
        }});
        assertEquals(expected, tensor.sum(new int[] {0}, true));

        expected = new Tensor(new double[][][] {{
                {14, 18, 21, 24}},
                {{26, 39, 16, 34},
                }});
        assertEquals(expected, tensor.sum(new int[] {1}, true));

        expected = new Tensor(new double[][][] {{
                {25}, {10}, {42}},
                {{40}, {11}, {64},
                }});
        assertEquals(expected, tensor.sum(new int[] {2}, true));

        expected = new Tensor(new double[][][] {{{40, 57, 37, 58}}});
        assertEquals(expected, tensor.sum(new int[] {0, 1}, true));

        expected = new Tensor(new double[][][] {{{77}}, {{115}}});
        assertEquals(expected, tensor.sum(new int[] {1, 2}, true));

        expected = new Tensor(new double[][][] {{{192.0}}});
        assertEquals(expected, tensor.sum(new int[] {0, 1, 2}, true));
    }

    @Test
    public void test_Prod_Scalar() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(5);
        assertEquals(expected, tensor.prod());
    }

    @Test
    public void test_Prod_1D() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(6);
        assertEquals(expected, tensor.prod());
    }

    @Test
    public void test_Prod_2D() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(40320);
        assertEquals(expected, tensor.prod());
    }

    @Test
    public void test_Prod_3D() {
        Tensor tensor = new Tensor(new double[][][]{
                {
                        {19, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12}
                },
                {
                        {13, 14, 15, 16},
                        {17, 18, 1, 20},
                        {21, 22, 23, 24},
                }
        });
        Tensor expected = new Tensor(6.204484017332394E23 );
        assertEquals(expected, tensor.prod());
    }

    @Test
    public void test_Prod_2D_View() {
        Tensor tensor = new Tensor(new int[]{3, 2, 1, 4, 5});
        Tensor view = JNDArray.broadcast(tensor, new int[]{4, 5});
        Tensor expected = new Tensor(207360000);
        assertEquals(expected, view.prod());
    }

    @Test
    public void test_Prod_1D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(new int[] {6});
        assertEquals(expected, tensor.prod(true));
    }

    @Test
    public void test_Prod_2D_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(new double[][] {{40320.0}});
        assertEquals(expected, tensor.prod(true));
    }

    @Test
    public void test_Prod_3D_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][][]{
                {
                        {19, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12}
                },
                {
                        {13, 14, 15, 16},
                        {17, 18, 1, 20},
                        {21, 22, 23, 24},
                }
        });
        Tensor expected = new Tensor(new double[][][] {{{6.204484017332394E23}}});
        assertEquals(expected, tensor.prod(true));
    }

    @Test
    public void test_Prod_1D_WithAxis() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(-5400);
        assertEquals(expected, tensor.prod(new int[] {0}));
    }

    @Test
    public void test_Prod_2D_WithAxis() {
        Tensor tensor = new Tensor(new double[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new double[]
                {0.0, -396.0, 276.0, -640.0}
        );
        assertEquals(expected, tensor.prod(new int[] {0}));

        expected = new Tensor(new double[]
                {-5400, 0, -21252.0}
        );
        assertEquals(expected, tensor.prod(new int[] {1}));

        expected = new Tensor(0.0);
        assertEquals(expected, tensor.prod(new int[] {0, 1}));
    }

    @Test
    public void test_Prod_3D_WithAxis() {
        Tensor tensor = new Tensor(new double[][][]{
                {
                        {4, 6, 7, 8},
                        {1, 2, 3, 4},
                        {9, 10, 11, 12}
                },
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        });
        Tensor expected = new Tensor(new double[][] {
                {20.0, 108.0, -21.0, 160.0},
                {0.0, -2.0, -12.0, 64.0},
                {189.0, 220.0, 253.0, -24.0}
        });
        assertEquals(expected, tensor.prod(new int[] {0}));

        expected = new Tensor(new double[][] {
                {36.0, 120.0, 231.0, 384.0},
                {0.0, -396.0, 276.0, -640.0},
        });
        assertEquals(expected, tensor.prod(new int[] {1}));

        expected = new Tensor(new double[][] {
                {1344.0, 24.0, 11880.0},
                {-5400.0, 0.0, -21252.0},
        });
        assertEquals(expected, tensor.prod(new int[] {2}));

        expected = new Tensor(new double[] {0, -47520, 63756, -245760});
        assertEquals(expected, tensor.prod(new int[] {0, 1}));

        expected = new Tensor(new double[] {3.8320128E8, 0.0});
        assertEquals(expected, tensor.prod(new int[] {1, 2}));

        expected = new Tensor(0.0);
        assertEquals(expected, tensor.prod(new int[] {0, 1, 2}));
    }

    @Test
    public void test_Prod_1D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(new double[] {-5400});
        assertEquals(expected, tensor.prod(new int[] {0}, true));
    }

    @Test
    public void test_Prod_2D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new double[][]
                {{0.0, -396.0, 276.0, -640.0}}
        );
        assertEquals(expected, tensor.prod(new int[] {0}, true));

        expected = new Tensor(new double[][]
                {{-5400}, {0}, {-21252.0}}
        );
        assertEquals(expected, tensor.prod(new int[] {1}, true));

        expected = new Tensor(new double[][]{{0}});
        assertEquals(expected, tensor.prod(new int[] {0, 1}, true));
    }

    @Test
    public void test_Prod_3D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][][]{
                {
                        {4, 6, 7, 8},
                        {1, 2, 3, 4},
                        {9, 10, 11, 12}
                },
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        });
        Tensor expected = new Tensor(new double[][][] {{
                {20.0, 108.0, -21.0, 160.0},
                {0.0, -2.0, -12.0, 64.0},
                {189.0, 220.0, 253.0, -24.0}
        }});
        assertEquals(expected, tensor.prod(new int[] {0}, true));

        expected = new Tensor(new double[][][] {{
                {36.0, 120.0, 231.0, 384.0}},
                {{0.0, -396.0, 276.0, -640.0},
                }});
        assertEquals(expected, tensor.prod(new int[] {1}, true));

        expected = new Tensor(new double[][][] {{
                {1344.0}, {24.0}, {11880.0}},
                {{-5400.0}, {0.0}, {-21252.0},
                }});
        assertEquals(expected, tensor.prod(new int[] {2}, true));

        expected = new Tensor(new double[][][] {{{0, -47520, 63756, -245760}}});
        assertEquals(expected, tensor.prod(new int[] {0, 1}, true));

        expected = new Tensor(new double[][][] {{{3.8320128E8}}, {{0.0}}});
        assertEquals(expected, tensor.prod(new int[] {1, 2}, true));

        expected = new Tensor(new double[][][] {{{0.0}}});
        assertEquals(expected, tensor.prod(new int[] {0, 1, 2}, true));
    }
}
