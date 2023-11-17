package internals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class TensorStatisticsTest {

    @Test
    public void test_Min_Scalar() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(5);
        assertEquals(expected, tensor.min());
    }

    @Test
    public void test_Min_1D() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(1);
        assertEquals(expected, tensor.min());
    }

    @Test
    public void test_Min_2D() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(1);
        assertEquals(expected, tensor.min());
    }

    @Test
    public void test_Min_3D() {
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
        Tensor expected = new Tensor(1);
        assertEquals(expected, tensor.min());
    }

    @Test
    public void test_Min_2D_View() {
        Tensor tensor = new Tensor(new int[]{3, 2, 1, 4, 5});
        Tensor view = JNDArray.broadcast(tensor, new int[]{4, 5});
        Tensor expected = new Tensor(1);
        assertEquals(expected, view.min());
    }

    @Test
    public void test_Max_Scalar() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(5);
        assertEquals(expected, tensor.max());
    }

    @Test
    public void test_Max_1D() {
        Tensor tensor = new Tensor(new int[] {2, 3, 1});
        Tensor expected = new Tensor(3);
        assertEquals(expected, tensor.max());
    }

    @Test
    public void test_Max_2D() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(8);
        assertEquals(expected, tensor.max());
    }

    @Test
    public void test_Max_3D() {
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
        Tensor expected = new Tensor(24);
        assertEquals(expected, tensor.max());
    }

    @Test
    public void test_Max_2D_View() {
        Tensor tensor = new Tensor(new int[]{3, 2, 1, 4, 5});
        Tensor view = JNDArray.broadcast(tensor, new int[]{4, 5});
        Tensor expected = new Tensor(5);
        assertEquals(expected, view.max());
    }

    @Test
    public void test_Min_Scalar_KeepDimensions() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(5);
        assertEquals(expected, tensor.min(true));
    }

    @Test
    public void test_Min_1D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(new int[] {1});
        assertEquals(expected, tensor.min(true));
    }

    @Test
    public void test_Min_2D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(new int[][] {{1}});
        assertEquals(expected, tensor.min(true));
    }

    @Test
    public void test_Min_3D_KeepDimensions() {
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
        Tensor expected = new Tensor(new int[][][] {{{1}}});
        assertEquals(expected, tensor.min(true));
    }

    @Test
    public void test_Max_Scalar_KeepDimensions() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(5);
        assertEquals(expected, tensor.max(true));
    }

    @Test
    public void test_Max_1D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(new int[] {3});
        assertEquals(expected, tensor.max(true));
    }

    @Test
    public void test_Max_2D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(new int[][] {{8}});
        assertEquals(expected, tensor.max(true));
    }

    @Test
    public void test_Max_3D_KeepDimensions() {
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
        Tensor expected = new Tensor(new int[][][] {{{24}}});
        assertEquals(expected, tensor.max(true));
    }

    @Test
    public void test_Min_1D_WithAxis() {
        Tensor tensor = new Tensor(new int[]
                        {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(-3);
        assertEquals(expected, tensor.min(new int[] {0}));
    }



    @Test
    public void test_Min_2D_WithAxis() {
        Tensor tensor = new Tensor(new int[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new int[]
                {0, -1, -4, -2}
        );
        assertEquals(expected, tensor.min(new int[] {0}));

        expected = new Tensor(new int[]
                {-3, -4, -2}
        );
        assertEquals(expected, tensor.min(new int[] {1}));

        expected = new Tensor(-4);
        assertEquals(expected, tensor.min(new int[] {0, 1}));
    }

    @Test
    public void test_Min_3D_WithAxis() {
        Tensor tensor = new Tensor(new int[][][]{
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
        Tensor expected = new Tensor(new int[][] {
                {4, 6, -3, 8},
                {0, -1, -4, 4},
                {9, 10, 11, -2}
        });
        assertEquals(expected, tensor.min(new int[] {0}));

        expected = new Tensor(new int[][] {
                {1, 2, 3, 4},
                {0, -1, -4, -2},
        });
        assertEquals(expected, tensor.min(new int[] {1}));

        expected = new Tensor(new int[][] {
                {4, 1, 9},
                {-3, -4, -2},
        });
        assertEquals(expected, tensor.min(new int[] {2}));

        expected = new Tensor(new int[] {0, -1, -4, -2});
        assertEquals(expected, tensor.min(new int[] {0, 1}));

        expected = new Tensor(new int[] {1, -4});
        assertEquals(expected, tensor.min(new int[] {1, 2}));

        expected = new Tensor(-4);
        assertEquals(expected, tensor.min(new int[] {0, 1, 2}));
    }

    @Test
    public void test_Min_1D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(new int[] {-3});
        assertEquals(expected, tensor.min(new int[] {0}, true));
    }

    @Test
    public void test_Min_2D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new int[][]
                {{0, -1, -4, -2}}
        );
        assertEquals(expected, tensor.min(new int[] {0}, true));

        expected = new Tensor(new int[][]
                {{-3}, {-4}, {-2}}
        );
        assertEquals(expected, tensor.min(new int[] {1}, true));

        expected = new Tensor(new int[][]{{-4}});
        assertEquals(expected, tensor.min(new int[] {0, 1}, true));
    }

    @Test
    public void test_Min_3D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][][]{
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
        Tensor expected = new Tensor(new int[][][] {{
                {4, 6, -3, 8},
                {0, -1, -4, 4},
                {9, 10, 11, -2}
        }});
        assertEquals(expected, tensor.min(new int[] {0}, true));

        expected = new Tensor(new int[][][] {{
                {1, 2, 3, 4}},
                {{0, -1, -4, -2},
        }});
        assertEquals(expected, tensor.min(new int[] {1}, true));

        expected = new Tensor(new int[][][] {{
                {4}, {1}, {9}},
                {{-3}, {-4}, {-2},
        }});
        assertEquals(expected, tensor.min(new int[] {2}, true));

        expected = new Tensor(new int[][][] {{{0, -1, -4, -2}}});
        assertEquals(expected, tensor.min(new int[] {0, 1}, true));

        expected = new Tensor(new int[][][] {{{1}}, {{-4}}});
        assertEquals(expected, tensor.min(new int[] {1, 2}, true));

        expected = new Tensor(new int[][][] {{{-4}}});
        assertEquals(expected, tensor.min(new int[] {0, 1, 2}, true));
    }

    @Test
    public void test_Max_1D_WithAxis() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(20);
        assertEquals(expected, tensor.max(new int[] {0}));
    }

    @Test
    public void test_Max_2D_WithAxis() {
        Tensor tensor = new Tensor(new int[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new int[]
                {21, 22, 23, 20}
        );
        assertEquals(expected, tensor.max(new int[] {0}));

        expected = new Tensor(new int[]
                {20, 16, 23}
        );
        assertEquals(expected, tensor.max(new int[] {1}));

        expected = new Tensor(23);
        assertEquals(expected, tensor.max(new int[] {0, 1}));
    }

    @Test
    public void test_Max_3D_WithAxis() {
        Tensor tensor = new Tensor(new int[][][]{
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
        Tensor expected = new Tensor(new int[][] {
                {5, 18, 7, 20},
                {1, 2, 3, 16},
                {21, 22, 23, 12}
        });
        assertEquals(expected, tensor.max(new int[] {0}));

        expected = new Tensor(new int[][] {
                {9, 10, 11, 12},
                {21, 22, 23, 20},
        });
        assertEquals(expected, tensor.max(new int[] {1}));

        expected = new Tensor(new int[][] {
                {8, 4, 12},
                {20, 16, 23},
        });
        assertEquals(expected, tensor.max(new int[] {2}));

        expected = new Tensor(new int[] {21, 22, 23, 20});
        assertEquals(expected, tensor.max(new int[] {0, 1}));

        expected = new Tensor(new int[] {12, 23});
        assertEquals(expected, tensor.max(new int[] {1, 2}));

        expected = new Tensor(23);
        assertEquals(expected, tensor.max(new int[] {0, 1, 2}));
    }

    @Test
    public void test_Max_1D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(new int[] {20});
        assertEquals(expected, tensor.max(new int[] {0}, true));
    }

    @Test
    public void test_Max_2D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new int[][]
                {{21, 22, 23, 20}}
        );
        assertEquals(expected, tensor.max(new int[] {0}, true));

        expected = new Tensor(new int[][]
                {{20}, {16}, {23}}
        );
        assertEquals(expected, tensor.max(new int[] {1}, true));

        expected = new Tensor(new int[][] {{23}});
        assertEquals(expected, tensor.max(new int[] {0, 1}, true));
    }

    @Test
    public void test_Max_3D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][][]{
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
        Tensor expected = new Tensor(new int[][][] {{
                {5, 18, 7, 20},
                {1, 2, 3, 16},
                {21, 22, 23, 12}
        }});
        assertEquals(expected, tensor.max(new int[] {0}, true));
        expected = new Tensor(new int[][][] {{
                {9, 10, 11, 12}},
                {{21, 22, 23, 20},
        }});
        assertEquals(expected, tensor.max(new int[] {1}, true));

        expected = new Tensor(new int[][][] {
                {
                        {8},
                        {4},
                        {12}
                },
                {
                        {20},
                        {16},
                        {23}
                }
        });
        assertEquals(expected, tensor.max(new int[] {2}, true));

        expected = new Tensor(new int[][][] {{{21, 22, 23, 20}}});
        assertEquals(expected, tensor.max(new int[] {0, 1}, true));

        expected = new Tensor(new int[][][] {
                {{12}},
                {{23}}
        });
        assertEquals(expected, tensor.max(new int[] {1, 2}, true));

        expected = new Tensor(new int[][][] {{{23}}});
        assertEquals(expected, tensor.max(new int[] {0, 1, 2}, true));
    }

    @Test
    public void test_ArgMin_Scalar() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(0);
        assertEquals(expected, tensor.argMin());
    }

    @Test
    public void test_ArgMin_1D() {
        Tensor tensor = new Tensor(new int[] {2, 0, 3});
        Tensor expected = new Tensor(1);
        assertEquals(expected, tensor.argMin());
    }

    @Test
    public void test_ArgMin_2D() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 0, 8},
        });
        Tensor expected = new Tensor(6);
        assertEquals(expected, tensor.argMin());
    }

    @Test
    public void test_ArgMin_3D() {
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
        Tensor expected = new Tensor(18);
        assertEquals(expected, tensor.argMin());
    }

    @Test
    public void test_ArgMin_2D_View() {
        Tensor tensor = new Tensor(new int[]{3, 2, 1, 4, 5});
        Tensor view = JNDArray.broadcast(tensor, new int[]{4, 5});
        Tensor expected = new Tensor(2);
        assertEquals(expected, view.argMin());
    }

    @Test
    public void test_ArgMax_Scalar() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(0);
        assertEquals(expected, tensor.argMax());
    }

    @Test
    public void test_ArgMax_1D() {
        Tensor tensor = new Tensor(new int[] {2, 3, 1});
        Tensor expected = new Tensor(1);
        assertEquals(expected, tensor.argMax());
    }

    @Test
    public void test_ArgMax_2D() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 8, 7},
        });
        Tensor expected = new Tensor(6);
        assertEquals(expected, tensor.argMax());
    }

    @Test
    public void test_ArgMax_3D() {
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
        Tensor expected = new Tensor(23);
        assertEquals(expected, tensor.argMax());
    }

    @Test
    public void test_ArgMax_2D_View() {
        Tensor tensor = new Tensor(new int[]{3, 2, 1, 4, 5});
        Tensor view = JNDArray.broadcast(tensor, new int[]{4, 5});
        Tensor expected = new Tensor(4);
        assertEquals(expected, view.argMax());
    }

    @Test
    public void test_ArgMin_Scalar_KeepDimensions() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(0);
        assertEquals(expected, tensor.argMin(true));
    }

    @Test
    public void test_ArgMin_1D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(new int[] {1});
        assertEquals(expected, tensor.argMin(true));
    }

    @Test
    public void test_ArgMin_2D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(new int[][] {{1}});
        assertEquals(expected, tensor.argMin(true));
    }

    @Test
    public void test_ArgMin_3D_KeepDimensions() {
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
        Tensor expected = new Tensor(new int[][][] {{{18}}});
        assertEquals(expected, tensor.argMin(true));
    }

    @Test
    public void test_ArgMax_Scalar_KeepDimensions() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(0);
        assertEquals(expected, tensor.argMax(true));
    }

    @Test
    public void test_ArgMax_1D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(new int[] {2});
        assertEquals(expected, tensor.argMax(true));
    }

    @Test
    public void test_ArgMax_2D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(new int[][] {{7}});
        assertEquals(expected, tensor.argMax(true));
    }

    @Test
    public void test_ArgMax_3D_KeepDimensions() {
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
        Tensor expected = new Tensor(new int[][][] {{{23}}});
        assertEquals(expected, tensor.argMax(true));
    }

    @Test
    public void test_ArgMin_1D_WithAxis() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(2);
        assertEquals(expected, tensor.argMin(0));
    }

    @Test
    public void test_ArgMin_2D_WithAxis() {
        Tensor tensor = new Tensor(new int[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new int[]
                {1, 1, 1, 2}
        );
        assertEquals(expected, tensor.argMin(0));

        expected = new Tensor(new int[]
                {2, 2, 3}
        );
        assertEquals(expected, tensor.argMin(1));
    }

    @Test
    public void test_ArgMin_3D_WithAxis() {
        Tensor tensor = new Tensor(new int[][][]{
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
        Tensor expected = new Tensor(new int[][] {
                {0, 0, 1, 0},
                {1, 1, 1, 0},
                {0, 0, 0, 1}
        });
        assertEquals(expected, tensor.argMin(0));

        expected = new Tensor(new int[][] {
                {1, 1, 1, 1},
                {1, 1, 1, 2},
        });
        assertEquals(expected, tensor.argMin(1));

        expected = new Tensor(new int[][] {
                {0, 0, 0},
                {2, 2, 3},
        });
        assertEquals(expected, tensor.argMin(2));
    }

    @Test
    public void test_ArgMax_1D_WithAxis() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(3);
        assertEquals(expected, tensor.argMax(0));
    }

    @Test
    public void test_ArgMax_2D_WithAxis() {
        Tensor tensor = new Tensor(new int[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new int[]
                {2, 2, 2, 0}
        );
        assertEquals(expected, tensor.argMax(0));

        expected = new Tensor(new int[]
                {3, 3, 2}
        );
        assertEquals(expected, tensor.argMax(1));
    }

    @Test
    public void test_ArgMax_3D_WithAxis() {
        Tensor tensor = new Tensor(new int[][][]{
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
        Tensor expected = new Tensor(new int[][] {
                {1, 1, 0, 1},
                {0, 0, 0, 1},
                {1, 1, 1, 0}
        });
        assertEquals(expected, tensor.argMax(0));

        expected = new Tensor(new int[][] {
                {2, 2, 2, 2},
                {2, 2, 2, 0},
        });
        assertEquals(expected, tensor.argMax(1));

        expected = new Tensor(new int[][] {
                {3, 3, 3},
                {3, 3, 2},
        });
        assertEquals(expected, tensor.argMax(2));
    }

    @Test
    public void test_ArgMin_1D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(new int[] {2});
        assertEquals(expected, tensor.argMin(0, true));
    }

    @Test
    public void test_ArgMin_2D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new int[][]
                {{1, 1, 1, 2}}
        );
        assertEquals(expected, tensor.argMin(0, true));

        expected = new Tensor(new int[][]
                {{2}, {2}, {3}}
        );
        assertEquals(expected, tensor.argMin(1, true));

    }

    @Test
    public void test_ArgMin_3D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][][]{
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
        Tensor expected = new Tensor(new int[][][] {{
                {0, 0, 1, 0},
                {1, 1, 1, 0},
                {0, 0, 0, 1}
        }});
        assertEquals(expected, tensor.argMin(0, true));

        expected = new Tensor(new int[][][] {{
                {1, 1, 1, 1}},
                {{1, 1, 1, 2},
                }});
        assertEquals(expected, tensor.argMin(1, true));

        expected = new Tensor(new int[][][] {{
                {0}, {0}, {0}},
                {{2}, {2}, {3},
                }});
        assertEquals(expected, tensor.argMin(2, true));
    }

    @Test
    public void test_ArgMax_1D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(new int[] {3});
        assertEquals(expected, tensor.argMax(0, true));
    }

    @Test
    public void test_ArgMax_2D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new int[][]
                {{2, 2, 2, 0}}
        );
        assertEquals(expected, tensor.argMax(0, true));

        expected = new Tensor(new int[][]
                {{3}, {3}, {2}}
        );
        assertEquals(expected, tensor.argMax(1, true));

    }

    @Test
    public void test_ArgMax_3D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new int[][][]{
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
        Tensor expected = new Tensor(new int[][][] {{
                {1, 1, 0, 1},
                {0, 0, 0, 1},
                {1, 1, 1, 0}
        }});
        assertEquals(expected, tensor.argMax(0, true));

        expected = new Tensor(new int[][][] {{
                {2, 2, 2, 2}},
                {{2, 2, 2, 0},
                }});
        assertEquals(expected, tensor.argMax(1, true));

        expected = new Tensor(new int[][][] {{
                {3}, {3}, {3}},
                {{3}, {3}, {2},
                }});
        assertEquals(expected, tensor.argMax(2, true));
    }

    @Test
    public void test_Mean_Scalar() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(5);
        assertEquals(expected, tensor.mean());
    }

    @Test
    public void test_Mean_1D() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(2);
        assertEquals(expected, tensor.mean());
    }

    @Test
    public void test_Mean_2D() {
        Tensor tensor = new Tensor(new double[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(4.5);
        assertEquals(expected, tensor.mean());
    }

    @Test
    public void test_Mean_3D() {
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
        Tensor expected = new Tensor(12.5);
        assertEquals(expected, tensor.mean());
    }

    @Test
    public void test_Mean_1D_KeepDimensions() {
        Tensor tensor = new Tensor(new int[] {2, 1, 3});
        Tensor expected = new Tensor(new int[] {2});
        assertEquals(expected, tensor.mean(true));
    }

    @Test
    public void test_Mean_2D_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(new double[][] {{4.5}});
        assertEquals(expected, tensor.mean(true));
    }

    @Test
    public void test_Mean_3D_KeepDimensions() {
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
        Tensor expected = new Tensor(new double[][][] {{{12.5}}});
        assertEquals(expected, tensor.mean(true));
    }

    @Test
    public void test_Mean_1D_WithAxis() {
        Tensor tensor = new Tensor(new int[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(10);
        assertEquals(expected, tensor.mean(new int[] {0}));
    }

    @Test
    public void test_Mean_2D_WithAxis() {
        Tensor tensor = new Tensor(new double[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new double[]
                {8.666666666666666,  13.0,  5.333333333333333,  11.333333333333334}
        );
        assertEquals(expected, tensor.mean(new int[] {0}));

        expected = new Tensor(new double[]
                {10, 2.75, 16}
        );
        assertEquals(expected, tensor.mean(new int[] {1}));

        expected = new Tensor(9.583333333333334);
        assertEquals(expected, tensor.mean(new int[] {0, 1}));
    }

    @Test
    public void test_Mean_3D_WithAxis() {
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
                {4.5, 12, 2, 14},
                {0.5, 0.5, -0.5, 10},
                {15, 16, 17, 5}
        });
        assertEquals(expected, tensor.mean(new int[] {0}));

        expected = new Tensor(new double[][] {
                {4.666666666666667, 6.0, 7.0, 8.0},
                {8.666666666666666, 13.0, 5.333333333333333, 11.333333333333334},
        });
        assertEquals(expected, tensor.mean(new int[] {1}));

        expected = new Tensor(new double[][] {
                {6.25, 2.5, 10.5},
                {10.0, 2.75, 16.0},
        });
        assertEquals(expected, tensor.mean(new int[] {2}));

        expected = new Tensor(new double[] {6.666666666666667, 9.5, 6.166666666666667, 9.666666666666666});
        assertEquals(expected, tensor.mean(new int[] {0, 1}));

        expected = new Tensor(new double[] {6.416666666666667,  9.583333333333334});
        assertEquals(expected, tensor.mean(new int[] {1, 2}));

        expected = new Tensor(8.0);
        assertEquals(expected, tensor.mean(new int[] {0, 1, 2}));
    }

    @Test
    public void test_Mean_1D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(new double[] {10});
        assertEquals(expected, tensor.mean(new int[] {0}, true));
    }

    @Test
    public void test_Mean_2D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new double[][]
                {{8.666666666666666,  13.0,  5.333333333333333,  11.333333333333334}}
        );
        assertEquals(expected, tensor.mean(new int[] {0}, true));

        expected = new Tensor(new double[][]
                {{10}, {2.75}, {16}}
        );
        assertEquals(expected, tensor.mean(new int[] {1}, true));

        expected = new Tensor(new double[][]{{9.583333333333334}});
        assertEquals(expected, tensor.mean(new int[] {0, 1}, true));
    }

    @Test
    public void test_Mean_3D_WithAxis_KeepDimensions() {
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
                {4.5, 12, 2, 14},
                {0.5, 0.5, -0.5, 10},
                {15, 16, 17, 5}
        }});
        assertEquals(expected, tensor.mean(new int[] {0}, true));

        expected = new Tensor(new double[][][] {{
                {4.666666666666667, 6.0, 7.0, 8.0}},
                {{8.666666666666666, 13.0, 5.333333333333333, 11.333333333333334},
                }});
        assertEquals(expected, tensor.mean(new int[] {1}, true));

        expected = new Tensor(new double[][][] {{
                {6.25}, {2.5}, {10.5}},
                {{10.0}, {2.75}, {16},
                }});
        assertEquals(expected, tensor.mean(new int[] {2}, true));

        expected = new Tensor(new double[][][] {{{6.666666666666667, 9.5, 6.166666666666667, 9.666666666666666}}});
        assertEquals(expected, tensor.mean(new int[] {0, 1}, true));

        expected = new Tensor(new double[][][] {{{6.416666666666667}}, {{9.583333333333334}}});
        assertEquals(expected, tensor.mean(new int[] {1, 2}, true));

        expected = new Tensor(new double[][][] {{{8.0}}});
        assertEquals(expected, tensor.mean(new int[] {0, 1, 2}, true));
    }

    @Test
    public void test_Std_Scalar() {
        Tensor tensor = new Tensor(5);
        Tensor expected = new Tensor(0.0);
        assertEquals(expected, tensor.std());
    }

    @Test
    public void test_Std_1D() {
        Tensor tensor = new Tensor(new double[] {2, 1, 3});
        Tensor expected = new Tensor(0.816496580927726);
        assertEquals(expected, tensor.std());
    }

    @Test
    public void test_Std_2D() {
        Tensor tensor = new Tensor(new double[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(2.29128784747792);
        assertEquals(expected, tensor.std());
    }

    @Test
    public void test_Std_3D() {
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
        Tensor expected = new Tensor(6.922186552431729);
        assertEquals(expected, tensor.std());
    }

    @Test
    public void test_Std_1D_KeepDimensions() {
        Tensor tensor = new Tensor(new double[] {2, 1, 3});
        Tensor expected = new Tensor(new double[] {0.816496580927726});
        assertEquals(expected, tensor.std(true));
    }

    @Test
    public void test_Std_2D_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][] {
                {2, 1, 3, 4},
                {5, 6, 7, 8},
        });
        Tensor expected = new Tensor(new double[][] {{2.29128784747792}});
        assertEquals(expected, tensor.std(true));
    }

    @Test
    public void test_Std_3D_KeepDimensions() {
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
        Tensor expected = new Tensor(new double[][][] {{{6.922186552431729}}});
        assertEquals(expected, tensor.std(true));
    }

    @Test
    public void test_Std_1D_WithAxis() {
        Tensor tensor = new Tensor(new double[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(9.460443964212251 );
        assertEquals(expected, tensor.std(new int[] {0}));
    }

    @Test
    public void test_Std_2D_WithAxis() {
        Tensor tensor = new Tensor(new double[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new double[]
                {8.9566858950296,  10.03327796219494,  12.498888839501783,  9.568466729604882}
        );
        assertEquals(expected, tensor.std(new int[] {0}));

        expected = new Tensor(new double[]
                {9.460443964212251,  7.790218225441442,  10.41633332799983}
        );
        assertEquals(expected, tensor.std(new int[] {1}));

        expected = new Tensor(10.75064597542347);
        assertEquals(expected, tensor.std(new int[] {0, 1}));
    }

    @Test
    public void test_Std_3D_WithAxis() {
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
                {0.5,  6.0,  5.0,  6.0},
                {0.5,  1.5,  3.5,  6.0},
                {6.0,  6.0,  6.0,  7.0}
        });
        assertEquals(expected, tensor.std(new int[] {0}));

        expected = new Tensor(new double[][] {
                {3.2998316455372216, 3.265986323710904, 3.265986323710904, 3.265986323710904},
                {8.9566858950296, 10.03327796219494, 12.498888839501783, 9.568466729604882},
        });
        assertEquals(expected, tensor.std(new int[] {1}));

        expected = new Tensor(new double[][] {
                {1.479019945774904, 1.118033988749895, 1.118033988749895},
                {9.460443964212251, 7.790218225441442, 10.41633332799983},
        });
        assertEquals(expected, tensor.std(new int[] {2}));

        expected = new Tensor(new double[] {7.039570693980958, 8.241156876717412, 9.172725270544676, 7.340905181848413});
        assertEquals(expected, tensor.std(new int[] {0, 1}));

        expected = new Tensor(new double[] {3.499007795869503, 10.75064597542347});
        assertEquals(expected, tensor.std(new int[] {1, 2}));

        expected = new Tensor(8.149642118931768);
        assertEquals(expected, tensor.std(new int[] {0, 1, 2}));
    }

    @Test
    public void test_Std_1D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[]
                {5, 18, -3, 20}
        );
        Tensor expected = new Tensor(new double[] {9.460443964212251});
        assertEquals(expected, tensor.std(new int[] {0}, true));
    }

    @Test
    public void test_Std_2D_WithAxis_KeepDimensions() {
        Tensor tensor = new Tensor(new double[][]
                {
                        {5, 18, -3, 20},
                        {0, -1, -4, 16},
                        {21, 22, 23, -2},
                }
        );
        Tensor expected = new Tensor(new double[][]
                {{8.9566858950296,  10.03327796219494,  12.498888839501783,  9.568466729604882}}
        );
        assertEquals(expected, tensor.std(new int[] {0}, true));

        expected = new Tensor(new double[][]
                {{9.460443964212251}, {7.790218225441442}, {10.41633332799983}}
        );
        assertEquals(expected, tensor.std(new int[] {1}, true));

        expected = new Tensor(new double[][]{{10.75064597542347}});
        assertEquals(expected, tensor.std(new int[] {0, 1}, true));
    }

    @Test
    public void test_Std_3D_WithAxis_KeepDimensions() {
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
                {0.5,  6.0,  5.0,  6.0},
                {0.5,  1.5,  3.5,  6.0},
                {6.0,  6.0,  6.0,  7.0}
        }});
        assertEquals(expected, tensor.std(new int[] {0}, true));

        expected = new Tensor(new double[][][] {{
                {3.2998316455372216, 3.265986323710904, 3.265986323710904, 3.265986323710904}},
                {{8.9566858950296, 10.03327796219494, 12.498888839501783, 9.568466729604882},
                }});
        assertEquals(expected, tensor.std(new int[] {1}, true));

        expected = new Tensor(new double[][][] {{
                {1.479019945774904}, {1.118033988749895}, {1.118033988749895}},
                {{9.460443964212251}, {7.790218225441442}, {10.41633332799983},
                }});
        assertEquals(expected, tensor.std(new int[] {2}, true));

        expected = new Tensor(new double[][][] {{{7.039570693980958, 8.241156876717412, 9.172725270544676, 7.340905181848413}}});
        assertEquals(expected, tensor.std(new int[] {0, 1}, true));

        expected = new Tensor(new double[][][] {{{3.499007795869503}}, {{10.75064597542347}}});
        assertEquals(expected, tensor.std(new int[] {1, 2}, true));

        expected = new Tensor(new double[][][] {{{8.149642118931768}}});
        assertEquals(expected, tensor.std(new int[] {0, 1, 2}, true));
    }
}
