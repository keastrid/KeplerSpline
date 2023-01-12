package com.astroimagej.bspline;

import com.astroimagej.bspline.util.Pair;
import com.astroimagej.bspline.util.Util;
import org.hipparchus.exception.MathIllegalArgumentException;
import org.hipparchus.linear.*;
import org.hipparchus.stat.descriptive.moment.StandardDeviation;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

//todo logging api so AIJ can replace the log messages
public class BSplineUtil {
    // checked, minor differences with last few row elements, differ by ~0.01
    public static Pair<RealVector, RealMatrix> choleskyBand(RealMatrix l, double minInf) {
        var bw = l.getRowDimension();
        var nn = l.getColumnDimension();

        var n = nn - bw;
        var t1 = IntStream.range(0, n).toArray();
        if (t1.length == 0) {
            t1 = new int[]{0};
        }
        var negative = l.getSubMatrix(new int[]{0}, t1)
                .mapToSelf(d -> Util.bool2int(d <= minInf));

        if (Util.isAny(negative, d -> d > 0) || !Util.isAll(l, Double::isFinite)) {
            System.out.println("Bad entries: ");//todo complete
            return new Pair<>(negative.getColumnVector(0), l);//todo column or row?
        }

        RealMatrix lower;
        try {
            lower = choleskyBanded(l.getSubMatrix(0, l.getRowDimension()-1, 0, n-1), EnumSet.noneOf(CholeskyBanded.class));
        } catch (MathIllegalArgumentException e) {
            e.printStackTrace();
            // Figure out where the error is
            lower = l.copy();
            var kn = bw - 1;
            var spot = IntStream.range(1, kn+1).toArray();

            for (int j = 0; j < n; j++) {
                lower.setEntry(0, j, Math.sqrt(lower.getEntry(0, j)));
                //todo check
                var t = lower.getSubMatrix(spot, new int[]{j}).scalarMultiply(1d/lower.getEntry(0, j));
                var p = 0;
                for (int i : spot) {
                    lower.setEntry(i, j, t.getEntry(p++, 0));
                }

                var x = lower.getSubMatrix(spot, new int[]{j});
                if (!Util.isAll(x, Double::isFinite)) {
                    System.out.println("Bad entries: ");//todo complete
                    return new Pair<>(MatrixUtils.createRealVector(new double[]{j}), l);
                }
            }
        }

        // Restore padding
        var L = MatrixUtils.createRealMatrix(l.getRowDimension(), l.getColumnDimension());
        L.setSubMatrix(lower.getData(), 0, 0);

        return new Pair<>(MatrixUtils.createRealVector(new double[]{-1}), L);
    }

    // Tested, works
    public static Pair<int[], RealMatrix> exofastCholeskyBand(RealMatrix lower, double minInf) {
        var bw = lower.getRowDimension();
        var nn = lower.getColumnDimension();
        var n = nn - bw;

        var negative = Util.predicateIndices(lower.getRowVector(0).getSubVector(0, n), d -> d <= minInf);
        if (negative.length > 0) {
            //System.out.println("bad entries" + Arrays.toString(negative));
            return new Pair<>(negative, null);
        }

        var kn = bw - 1;
        var spot = IntStream.range(1, kn+1).toArray();
        var bi = MatrixUtils.createRealVector(IntStream.range(0, kn).asDoubleStream().toArray());

        for (int i = 1; i <= kn - 1; i++) {
            int finalI = i;
            bi = bi.append(MatrixUtils.createRealVector(IntStream.range(0, kn-i).map(e -> e + (kn+1) * finalI).asDoubleStream().toArray()));
        }

        var fL = new Util.FlatMap(lower);
        var t = fL.matrix();
        for (int j = 0; j <= n - 1; j++) {
            lower.setEntry(0, j, Math.sqrt(lower.getEntry(0, j)));
            for (int i : spot) {
                lower.setEntry(i, j, lower.getEntry(i, j) / lower.getEntry(0, j));
            }
            var x = lower.getSubMatrix(spot, new int[]{j});

            if (Util.isAny(x, d -> !Double.isFinite(d))) {
                return new Pair<>(new int[]{j}, lower);
            }

            var hmm = x.multiplyTransposed(x);
            var fHmm = new Util.FlatMap(hmm);
            var here = bi.mapAdd((j+1)*bw);

            for (int i = 0; i < here.getDimension(); i++) {
                fL.set((int) here.getEntry(i), fL.get((int) here.getEntry(i)) - fHmm.get((int) bi.getEntry(i)));
            }
        }

        return new Pair<>(new int[]{-1}, lower);
    }

    public static RealMatrix choleskySolve(RealMatrix a, RealMatrix bb) {
        var bw = a.getRowDimension();
        var n = bb.getRowDimension() - bw;
        var x = MatrixUtils.createRealMatrix(bb.getRowDimension(), bb.getColumnDimension());

        x.setSubMatrix(choSolveBanded(a.getSubMatrix(0, a.getRowDimension()-1, 0, n-1),
                bb.getSubMatrix(0, n-1, 0, bb.getColumnDimension()-1)).getData(),
                0, 0);

        return x;
    }

    //tested, only works for sq. matrix - can't use
    public static RealMatrix choSolveBanded(RealMatrix a, RealMatrix b) {//todo options control
        var checkFinite = true;
        var cb = a;

        // In python this converts to an ndarray, but we have saner typing so just recreate the finite check
        if (checkFinite) {
            if (!Util.isAll(cb, Double::isFinite)) {
                throw new IllegalStateException("Matrix must be finite");
            }
        }

        //return new QRDecomposition(cb.transpose()).getSolver().solve(b);
        return new CholeskyDecomposition(cb).getSolver().solve(b);
    }

    //tested, works
    public static RealVector exoFastCholeskySolve(RealMatrix a, RealVector b) {//todo cleanup
        var bw = a.getRowDimension();
        var n = b.getDimension() - bw;
        var kd = bw - 1;

        // First round
        for (int j = 0; j <= n - 1; j++) {
            b.setEntry(j, b.getEntry(j) / a.getEntry(0, j));
            int finalJ = j;
            IntStream.range(1, kd+1).forEach(spot -> {
                b.setEntry(finalJ+spot, b.getEntry(finalJ+spot) - b.getEntry(finalJ) * a.getEntry(spot, finalJ));
            });
        }

        // Second round
        for (int j = n-1; j >= 0; j--) {
            int finalJ = j;
            var spots = IntStream.range(0, kd).map(i -> kd - i).toArray();

            var is = Arrays.stream(spots).map(s -> s + finalJ).toArray();
            var b2 = MatrixUtils.createRealVector(spots.length);
            for (int i = 0; i < spots.length; i++) {
                b2.setEntry(i, b.getEntry(is[i]));
            }

            var l = a.getSubMatrix(spots, new int[]{j}).getColumnVector(0).ebeMultiply(b2);
            var total = Arrays.stream(l.toArray()).sum();
            b.setEntry(finalJ, (b.getEntry(finalJ) - total) / a.getEntry(0, finalJ));
        }

        return b;
    }

    public static RealMatrix choleskyBanded(RealMatrix ab, EnumSet<CholeskyBanded> options) {
        var overwriteAb = options.contains(CholeskyBanded.OVERWRITE_AB);
        var lower = options.contains(CholeskyBanded.LOWER);
        var checkFinite = !options.contains(CholeskyBanded.DO_NOT_CHECK_FINITE);

        // In python this converts to an ndarray, but we have saner typing so just recreate the finite check
        if (checkFinite) {
            if (!Util.isAll(ab, Double::isFinite)) {
                throw new IllegalStateException("Matrix must be finite");
            }
        }

        //todo python calls to c/fortran lib LAPACK for pbtrf, check matches
        //todo make it a lower matrix?
        return new CholeskyDecomposition(ab).getL(); //todo LT?
    }

    //tested, but no points rejected
    public static Pair<RealMatrix, Boolean> djsReject(RealMatrix data, RealMatrix model, RealMatrix outMask, RealMatrix inMask,
                                 /*RealMatrix sigma,*/ RealMatrix invVar, Double lower, Double upper) { //todo other options
        Double sigma = null;
        Double maxDev = null;
        RealVector maxRej = null;
        RealVector groupDim = null;
        RealVector groupSize = null;
        Boolean groupBadPix = null;
        int grow = 0;
        Boolean sticky = null;

        // Create outMask setting = 1 for good data
        if (outMask == null) {
            outMask = MatrixUtils.createRealMatrix(data.getRowDimension(), data.getColumnDimension());
        } else {
            if (!Util.shapeMatches(data, outMask)) {
                throw new IllegalArgumentException("Dimensions of data and outmask do not agree.");
            }
        }

        // Check other inputs
        if (model == null) {
            if (inMask != null) {
                outMask = inMask;
            }
            return new Pair<>(outMask, false);
        } else {
            if (!Util.shapeMatches(data, model)) {
                throw new IllegalArgumentException("Dimensions of data and model do not agree.");
            }
        }

        if (inMask != null) {
            if (!Util.shapeMatches(data, inMask)) {
                throw new IllegalArgumentException("Dimensions of data and inmask do not agree.");
            }
        }

        if (maxRej != null) {
            if (groupDim != null) {
                if (maxRej.getDimension() != groupDim.getDimension()) {
                    throw new IllegalArgumentException("maxrej and groupdim must have the same number of elements");
                }
            } else {
                groupDim = MatrixUtils.createRealVector(0);
            }
            if (groupSize != null) {
                if (maxRej.getDimension() != groupSize.getDimension()) {
                    throw new IllegalArgumentException("maxrej and groupdim must have the same number of elements");
                }
            } else {
                groupSize = MatrixUtils.createRealVector(0);
                groupSize = groupSize.append(data.getRowDimension());
            }
        }

        RealMatrix finalOutMask = outMask;

        // In python, this contains the indices that are good.
        // For us, there is no single index mapping, so we have a matrix of equal size that is true for the
        // good elements.
        RealMatrix iGood;

        var diff = data.subtract(model);

        if (sigma != null && invVar == null) {
            if (inMask != null) {
                iGood = MatrixUtils.createRealMatrix(inMask.getRowDimension(), inMask.getColumnDimension());
                iGood.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        // inMask and outMask should only contain 0 or 1
                        value = Util.bool2int(Util.int2bool((int) inMask.getEntry(row, column)) && Util.int2bool((int) finalOutMask.getEntry(row, column)));
                        return value;
                    }
                });
            } else {
                iGood = outMask.copy();
            }

            sigma = 0D;
            if (Util.isAny(iGood, d -> d > 1)) {
                var std = new StandardDeviation();
                iGood.walkInOptimizedOrder(new DefaultRealMatrixPreservingVisitor() {
                    @Override
                    public void visit(int row, int column, double value) {
                        if (value > 0) {
                            std.increment(diff.getEntry(row, column));//todo this can have issues
                        }
                    }
                });
                sigma = std.getResult();
            }
        }

        Double finalSigma = sigma;

        // The working array is badness, which is set to 0 for goo points (or already rejected points),
        // and positive values for bad points. The values determine just how bad a point is,
        // either corresponding to the number of sigma above or below the fit, or to the number of multiples of
        // maxDev away from the fit.
        var badness = MatrixUtils.createRealMatrix(outMask.getRowDimension(), outMask.getColumnDimension());
        var qBad = MatrixUtils.createRealMatrix(outMask.getRowDimension(), outMask.getColumnDimension());

        // Decide how bad a point is according to lower.
        if (lower != null) {
            if (sigma != null) {
                qBad.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        return Util.bool2int(diff.getEntry(row, column) < (-lower * finalSigma));
                    }
                });
                badness.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        return value + Util.bool2int((-diff.getEntry(row, column)/
                                (finalSigma + Util.bool2int(finalSigma == 0))) > 0) * qBad.getEntry(row, column);
                    }
                });
            } else {
                qBad.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        return Util.bool2int(diff.getEntry(row, column) * Math.sqrt(invVar.getEntry(row, column)) < -lower);
                    }
                });
                badness.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        return value + Util.bool2int(-diff.getEntry(row, column)/
                                (Math.sqrt(invVar.getEntry(row, column))) > 0) * qBad.getEntry(row, column);
                    }
                });
            }
        }

        // Decide how bad a point is according to upper.
        if (upper != null) {
            if (sigma != null) {
                qBad.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        return Util.bool2int(diff.getEntry(row, column) > (upper * finalSigma));
                    }
                });
                badness.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        return value + Util.bool2int((diff.getEntry(row, column)/
                                (finalSigma + Util.bool2int(finalSigma == 0))) > 0) * qBad.getEntry(row, column);
                    }
                });
            } else {
                qBad.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        return Util.bool2int(diff.getEntry(row, column) * Math.sqrt(invVar.getEntry(row, column)) > upper);
                    }
                });
                badness.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        return value + Util.bool2int(diff.getEntry(row, column)/
                                (Math.sqrt(invVar.getEntry(row, column))) > 0) * qBad.getEntry(row, column);
                    }
                });
            }
        }

        // Decide how bad a point is according to upper.
        if (maxDev != null) {
            qBad.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return Util.bool2int(Math.abs(diff.getEntry(row, column)) > maxDev);
                }
            });
            badness.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return value + ((Math.abs(diff.getEntry(row, column)) / maxDev) * qBad.getEntry(row, column));
                }
            });
        }

        // Do not consider rejecting points that are already rejected by inMask
        // Do not consider rejecting points that are already rejected by outMask, if sticky is set
        if (inMask != null) {
            badness.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return value * inMask.getEntry(row, column);
                }
            });
        }
        if (Boolean.TRUE.equals(sticky)) {
            badness.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return value * finalOutMask.getEntry(row, column);
                }
            });
        }

        // Reject a maximum of maxRej (additional) points in all the data, or
        // in each group as specified by groupSize, and optionally along each dimension specified by groupDim
        /*if (maxRej != null) { //todo ln347 yData = data
            // Loop over each dimension of groupDim or loop once if not set.
            for (int iLoop = 0; iLoop < Math.max(groupDim.getDimension(), 1); iLoop++) {
                // Assign an index number in this dimension to each datapoint
                RealMatrix dimNum;
                if (groupDim.getDimension() > 0) {
                    var ynDim = ;
                } else {
                    dimNum = MatrixUtils.createRealMatrix(1, 1);
                }
                
                // Loop over each vector specified by groupDim. 
                // For example, if this is a 2D array with groupDim=1, then loop over each column of data. 
                // If groupDim=2, then loop over each row.
                // If groupDim is not set, then use the whole image.
                for (int iVec = 0; iVec < Util.max(dimNum); iVec++) {
                    // At this point it is not possible that dimNum is not set
                    var indx =
                }
            }
        }*/

        // Now modify outMask, rejecting points spcified by inMask=0, outmask=0 if stick is set, or badness > 0
        var newMask = badness.copy();
        newMask.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value) {
                return value == 0 ? 1 : 0;
            }
        });

        if (grow > 0) {
            var rejects = newMask.copy();
            rejects.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return value == 0 ? 1 : 0;
                }
            });

            if (Util.isAny(rejects, d -> d > 0)) {
                //todo ln429
            }
        }

        if (inMask != null) {
            newMask.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return Util.bool2int(Util.int2bool(((int) value)) & Util.int2bool((int) inMask.getEntry(row, column)));
                }
            });
        }

        if (Boolean.TRUE.equals(sticky)) {
            newMask.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return Util.bool2int(Util.int2bool(((int) value)) & Util.int2bool((int) finalOutMask.getEntry(row, column)));
                }
            });
        }

        // Set qDone if the input mask is identical to the output mask;
        var qDone = newMask.equals(outMask);
        outMask = newMask;

        return new Pair<>(outMask, qDone);
    }

    //tested
    public static RealMatrix djsLaxisnum(int[] dims, Integer iAxis) {
        if (iAxis == null) iAxis = 0;

        var nDimen = dims.length;
        var result = MatrixUtils.createRealMatrix(dims[0], dims[1]); //todo order, check of size

        switch (nDimen) {
            case 1:
                break;
            case 2:
                if (iAxis == 0) {
                    IntStream.range(0, dims[0]).forEach(k -> {
                        result.setRow(k, DoubleStream.generate(() -> k).limit(dims[1]).toArray());
                    });
                } else if (iAxis == 1) {
                    IntStream.range(0, dims[1]).forEach(k -> {
                        result.setColumn(k, DoubleStream.generate(() -> k).limit(dims[0]).toArray());
                    });
                } else {
                    throw new IllegalStateException("Unexpected value: " + nDimen);
                }
                break;
            case 3: //todo 3D matrices
                switch (iAxis) {
                    case 0:
                        break;
                    case 1:
                        break;
                    case 2:
                        break;
                    default:
                        throw new IllegalStateException("Unexpected value: " + nDimen);
                }
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + nDimen);
        }

        return result;
    }

    public enum CholeskyBanded {
        OVERWRITE_AB,
        LOWER,
        DO_NOT_CHECK_FINITE
    }
}
