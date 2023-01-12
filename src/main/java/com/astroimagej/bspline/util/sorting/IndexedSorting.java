package com.astroimagej.bspline.util.sorting;

import org.hipparchus.linear.RealVector;
import org.hipparchus.linear.RealVectorPreservingVisitor;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

public class IndexedSorting<T extends Comparable<T>> implements Comparator<Integer> {
    private final T[] array;

    private IndexedSorting(T[] array) {
        this.array = array;
    }

    private Integer[] unsortedIndices() {
        return IntStream.range(0, array.length).boxed().toArray(Integer[]::new);
    }

    @Override
    public int compare(Integer index1, Integer index2) {
        return array[index1].compareTo(array[index2]);
    }

    /**
     * @return indices of sorted set in ascending order
     */
    public static <T extends Comparable<T>> Integer[] sort(T[] a) {
        var c = new IndexedSorting<T>(a);
        var idx = c.unsortedIndices();
        Arrays.parallelSort(idx, c);

        return idx;
    }

    /**
     * @return indices of sorted set in ascending order
     */
    public static int[] sort(RealVector v) {//todo this could be better
        var a = new Double[v.getDimension()];
        v.walkInOptimizedOrder(new RealVectorPreservingVisitor() {
            @Override
            public void start(int dimension, int start, int end) {
            }

            @Override
            public void visit(int index, double value) {
                a[index] = value;
            }

            @Override
            public double end() {
                return 0;
            }
        });

        var c = new IndexedSorting<Double>(a);
        var idx = c.unsortedIndices();
        Arrays.parallelSort(idx, c);

        return Arrays.stream(idx).mapToInt(i -> i).toArray();
    }
}
