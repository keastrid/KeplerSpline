package com.astroimagej.bspline.util;

import org.hipparchus.linear.*;

import java.util.Arrays;
import java.util.Collection;
import java.util.function.BiFunction;
import java.util.function.DoublePredicate;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class Util {
    public static int bool2int(boolean b) {
        return b ? 1 : 0;
    }

    public static boolean int2bool(int i) {
        return i > 0;
    }

    public static boolean isAny(RealMatrix m, DoublePredicate predicate) {
        for (double[] mDatum : m.getData()) {
            for (double v : mDatum) {
                if (predicate.test(v)) {
                    return true;
                }
            }
        }

        return false;
    }

    public static boolean isAny(RealVector v, DoublePredicate predicate) {
        for (double e : v.toArray()) {
            if (predicate.test(e)) {
                return true;
            }
        }
        return false;
    }

    public static boolean isAll(RealMatrix m, DoublePredicate predicate) {
        for (double[] mDatum : m.getData()) {
            for (double v : mDatum) {
                if (!predicate.test(v)) {
                    return false;
                }
            }
        }

        return true;
    }

    public static <T> boolean isAll(Collection<T> m, Predicate<T> predicate) {
        for (T t : m) {
            if (!predicate.test(t))
                return false;
        }

        return true;
    }

    public static RealVector and(RealVector a, RealVector b) {
        var o = a.copy();
        o.walkInOptimizedOrder(new DefaultVectorChangingVisitor() {
            @Override
            public double visit(int index, double value) {
                return (int) value & (int) b.getEntry(index);
            }
        });

        return o;
    }

    public static boolean shapeMatches(RealMatrix a, RealMatrix b) {
        return a.getRowDimension() == b.getRowDimension() && a.getColumnDimension() == b.getColumnDimension();
    }

    public static boolean shapeMatches(RealVector a, RealVector b) {
        return a.getDimension() == b.getDimension();
    }

    public static double max(RealMatrix a) {
        return a.walkInOptimizedOrder(new DefaultRealMatrixPreservingVisitor() {
            double out = Double.NaN;
            @Override
            public void visit(int row, int column, double value) {
                if (value > out) {
                    out = value;
                }
            }

            @Override
            public double end() {
                return out;
            }
        });
    }

    public static double min(RealMatrix a) {
        return a.walkInOptimizedOrder(new DefaultRealMatrixPreservingVisitor() {
            double out = Double.NaN;
            @Override
            public void visit(int row, int column, double value) {
                if (value < out) {
                    out = value;
                }
            }

            @Override
            public double end() {
                return out;
            }
        });
    }

    public static double max(RealVector a) {
        return a.walkInOptimizedOrder(new RealVectorPreservingVisitor() {
            double out = Double.NaN;

            @Override
            public void start(int dimension, int start, int end) {
            }

            @Override
            public void visit(int index, double value) {
                if (value > out) {
                    out = value;
                }
            }

            @Override
            public double end() {
                return out;
            }
        });
    }

    public static double min(RealVector a) {
        return a.walkInOptimizedOrder(new RealVectorPreservingVisitor() {
            double out = Double.NaN;

            @Override
            public void start(int dimension, int start, int end) {
            }

            @Override
            public void visit(int index, double value) {
                if (value < out) {
                    out = value;
                }
            }

            @Override
            public double end() {
                return out;
            }
        });
    }

    /**
     * @return <a href="https://pages.physics.wisc.edu/~craigm/idl/archive/msg00456.html">IDL's version of variance.</a>
     */
    public static double var(RealVector v) {
        var a = v.toArray();

        var stats = DoubleStream.of(a).summaryStatistics();
        return DoubleStream.of(a).map(v_i -> v_i - stats.getAverage())
                .map(d -> d*d).sum() / (double) (v.getDimension()-1);
    }

    public static double std(RealVector v) {
        var mean = mean(v.toArray());
        return Math.sqrt(mean(v.map(d -> Math.pow(Math.abs(d - mean), 2)).toArray()));
    }

    public static double[] logSpace(double start, double stop, int num) {
        // Generate spacings
        var pts = new double[num];
        var slope = (stop - start) / (num - 1);

        for (int i = 0; i < num; i++) {
            pts[i] = Math.pow(10, start + (slope * i));
        }

        pts[num-1] = Math.pow(10, stop);

        return pts;
    }

    public static RealVector takeIndices(RealVector v, int[] idx) {
        var o = MatrixUtils.createRealVector(idx.length);
        var p = 0;
        for (int i : idx) {
            o.setEntry(p++, v.getEntry(i));
        }

        return o;
    }

    public static RealVector takeIndices(int[] v, int[] idx) {
        var o = MatrixUtils.createRealVector(idx.length);
        var p = 0;
        for (int i : idx) {
            o.setEntry(p++, v[i]);
        }

        return o;
    }

    public static RealVector takeIndices(RealVector v, RealVector idx) {
        var o = MatrixUtils.createRealVector(idx.getDimension());
        var p = 0;
        for (double i : idx.toArray()) {
            o.setEntry(p++, v.getEntry((int) i));
        }

        return o;
    }

    public static int[] filterIndices(int[] idx, RealVector mask) {
        return IntStream.range(0, idx.length).filter(i -> mask.getEntry(i) >= 1).map(i -> idx[i]).toArray();
    }

    public static int[] nonZero(RealVector v) {
        return IntStream.range(0, v.getDimension()).filter(i -> v.getEntry(i) != 0).toArray();
    }

    public static int[] nonZero(int[] v) {
        return IntStream.range(0, v.length).filter(i -> v[i] != 0).toArray();
    }

    public static RealVector takeMask(RealVector v, RealVector mask) {
        return MatrixUtils.createRealVector(IntStream.range(0, v.getDimension())
                .filter(i -> mask.getEntry(i) >= 1).mapToDouble(v::getEntry).toArray());
    }

    public static RealVector takeIndices(RealMatrix m, int[] idx) {
        //todo not needed, but is a fast path we aren't using
        if (m.getRowDimension() == 1) {
            return takeIndices(flatten(m, Direction.ROW), idx);
        } else if (m.getColumnDimension() == 1) {
            return takeIndices(flatten(m, Direction.COLUMN), idx);
        } else {
            throw new IllegalArgumentException("Must be vectorizable");
        }
    }

    public static RealVector flatten(RealMatrix m, Direction d) {
        var v = MatrixUtils.createRealVector(m.getColumnDimension() * m.getRowDimension());
        var p = 0;
        for (int i = 0; i < d.dimension.apply(m); i++) {
            for (int j = 0; j < d.other().dimension.apply(m); j++) {
                v.setEntry(p++, d.entry.apply(m).apply(i, j));
            }
        }

        return v;
    }

    public static double median(RealVector v) {
        var l = v.getDimension();
        var a = v.toArray();
        Arrays.sort(a);
        if (l % 2 == 0) {
            return mean(a[l/2], a[(l/2)-1]);
        } else {
            return a[l/2];
        }
    }

    public static double mean(double... vs) {
        return Arrays.stream(vs).average().orElse(0);
    }

    public static double sum(RealVector v) {
        return v.walkInOptimizedOrder(new RealVectorPreservingVisitor() {
            double sum = 0;
            @Override
            public void start(int dimension, int start, int end) {
            }

            @Override
            public void visit(int index, double value) {
                sum += value;
            }

            @Override
            public double end() {
                return sum;
            }
        });
    }

    public static RealMatrix reshape(RealMatrix m, int rs, int cs) {
        var fm = new FlatMap(m);
        var o = new FlatMap(MatrixUtils.createRealMatrix(rs, cs));

        for (int i = 0; i < fm.size(); i++) {
            o.set(i, fm.get(i));
        }

        return o.matrix;
    }

    public static RealMatrix reshape(RealVector v, int rs, int cs) {
        var o = new FlatMap(MatrixUtils.createRealMatrix(rs, cs));

        for (int i = 0; i < v.getDimension(); i++) {
            o.set(i, v.getEntry(i));
        }

        return o.matrix;
    }

    public static RealMatrix vector2Matrix(RealVector v) {
        return MatrixUtils.createColumnRealMatrix(v.toArray());
    }

    //tested, works
    public static int[] uniq(RealVector x, int[] indices) {
        RealVector sorted;
        if (indices == null) {
            sorted = x;
        } else {
            sorted = takeIndices(x, indices);
        }

        return IntStream.range(0, sorted.getDimension()) // Generate all indices of the vector
                // compare 2 continuous elements of vector, comparing first element with the last one
                .filter(i -> sorted.getEntry(i) != sorted.getEntry(Math.floorMod(i+1, sorted.getDimension())))
                .toArray();
    }

    public static RealVector where(DoublePredicate condition, RealVector source, RealVector x, RealVector y) {
        var out = source.copy();
        out.walkInOptimizedOrder(new DefaultVectorChangingVisitor() {
            @Override
            public double visit(int index, double value) {
                return condition.test(value) ? x.getEntry(index) : y.getEntry(index);
            }
        });

        return out;
    }

    public static RealVector reverse(RealVector v) {
        var o = v.copy();
        var p = v.getDimension() - 1;
        for (int i = 0; i < v.getDimension(); i++) {
            o.setEntry(p--, v.getEntry(i));
        }

        return o;
    }

    public static int[] predicateIndices(RealVector v, DoublePredicate p) {
        return IntStream.range(0, v.getDimension()).filter(i -> p.test(v.getEntry(i))).toArray();
    }

    public static RealMatrix ebeMultiply(RealMatrix a, RealMatrix b) {
        if (!shapeMatches(a, b)) {
            throw new IllegalArgumentException("Shapes must match");
        }
        var o = a.copy();
        o.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value) {
                return value * b.getEntry(row, column);
            }
        });

        return o;
    }

    public static RealVector concatenate(Collection<RealVector> vs) {
        var o = MatrixUtils.createRealVector(0);
        for (RealVector v : vs) {
            o = o.append(v);
        }

        return o;
    }

    public static RealVector diff(RealVector v) {
        return MatrixUtils.createRealVector(IntStream.range(0, v.getDimension() - 1)
                .mapToDouble(i -> v.getEntry(i+1) - v.getEntry(i)).toArray());
    }

    public record TransposeFlatMap(RealMatrix matrix) {
        public void set(int i, double val) {
            matrix.setEntry(getRow(i), getColumn(i), val);
        }

        public double get(int i) {
            return matrix.getEntry(getRow(i), getColumn(i));
        }

        public int getRow(int i) {
            if (i >= matrix.getRowDimension() * matrix.getColumnDimension()) {
                throw new IndexOutOfBoundsException(i);
            }
            return i % matrix.getRowDimension();
        }

        public int getColumn(int i) {
            if (i >= matrix.getRowDimension() * matrix.getColumnDimension()) {
                throw new IndexOutOfBoundsException(i);
            }
            return i / matrix.getRowDimension();
        }
    }
    public record FlatMap(RealMatrix matrix) {
        public void set(int i, double val) {
            matrix.setEntry(getRow(i), getColumn(i), val);
        }

        public double get(int i) {
            return matrix.getEntry(getRow(i), getColumn(i));
        }

        public int getRow(int i) {
            if (i >= size()) {
                throw new IndexOutOfBoundsException(i);
            }
            return i % matrix.getRowDimension();
        }

        public int getColumn(int i) {
            if (i >= size()) {
                throw new IndexOutOfBoundsException(i);
            }
            return i / matrix.getRowDimension();
        }

        public int size() {
            return matrix.getRowDimension() * matrix.getColumnDimension();
        }
    }

    public enum Direction {
        ROW(RealMatrix::getRowDimension, m -> m::getEntry) {
            public Direction other() {
                return COLUMN;
            }
        },
        COLUMN(RealMatrix::getColumnDimension, m -> (i, j) -> m.getEntry(j, i)) {
            public Direction other() {
                return ROW;
            }
        };

        private final Function<RealMatrix, Integer> dimension;
        private final Function<RealMatrix, BiFunction<Integer, Integer, Double>> entry;

        Direction(Function<RealMatrix, Integer> dimension, Function<RealMatrix, BiFunction<Integer, Integer, Double>> entry) {
            this.dimension = dimension;
            this.entry = entry;
        }

        public abstract Direction other();
    }
}
