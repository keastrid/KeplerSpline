package com.astroimagej.bspline;

import com.astroimagej.bspline.util.DefaultVectorChangingVisitor;
import com.astroimagej.bspline.util.Pair;
import com.astroimagej.bspline.util.Triple;
import com.astroimagej.bspline.util.Util;
import com.astroimagej.bspline.util.sorting.IndexedSorting;
import org.hipparchus.linear.*;

import java.util.Arrays;
import java.util.stream.IntStream;

public class BSpline {
    //todo replace funcName with method reference
    //todo types
    RealVector breakpoints;
    int nOrd;
    int nPoly;
    RealVector mask;
    RealMatrix coeff;
    RealMatrix icoeff;
    double xMin;
    double xMax;
    String funcName;

    BSpline(RealVector x, int nOrd, int nPoly, RealVector bkpt, double bkSpread, BSplineArgs kArgs) {
        var xStats = Arrays.stream(x.toArray()).summaryStatistics();
        // Set the breakpoints
        if (bkpt == null) {
            var startX = xStats.getMin();
            var rangeX = xStats.getMax() - startX;

            //todo if placed in kargs
            int nBkpts;
            if (kArgs.bkSpace != null) {
                nBkpts = ((int) (rangeX/kArgs.bkSpace)) + 1;
                if (nBkpts < 2) {
                    nBkpts = 2;
                }
                var tempBkSpace = rangeX/(double) (nBkpts-1);
                //todo check range
                bkpt = MatrixUtils.createRealVector(IntStream.range(0, nBkpts).mapToDouble(i -> i*tempBkSpace + startX).toArray());
            }
            //todo if everyN

            var t = IndexedSorting.sort(bkpt);
            var iMin = t[0];
            var iMax = t[t.length-1];
            if (xStats.getMin() < bkpt.getEntry(iMin)) {
                bkpt.setEntry(iMin, xStats.getMin());
            }
            if (xStats.getMax() > bkpt.getEntry(iMax)) {
                bkpt.setEntry(iMax, xStats.getMax());
            }

            var nShortBkpt = bkpt.getDimension();
            var fullBkpt = bkpt.copy();

            double bkSpace;
            if (nShortBkpt == 1) {
                bkSpace = bkSpread;
            } else {
                bkSpace = (bkpt.getEntry(1) - bkpt.getEntry(0)) * bkSpread;
            }

            for (int i = 1; i < nOrd; i++) {
                var t1 = MatrixUtils.createRealVector(new double[]{bkpt.getEntry(0)-bkSpace*i});
                fullBkpt = t1.append(fullBkpt);
                fullBkpt = fullBkpt.append(bkpt.getEntry(nShortBkpt-1)+bkSpace*i);
            }

            // Set the attributes
            var nc = fullBkpt.getDimension() - nOrd;
            breakpoints = fullBkpt;
            this.nOrd = nOrd;
            this.nPoly = nPoly;
            var t2 = new double[fullBkpt.getDimension()];
            Arrays.fill(t2, 1);
            mask = MatrixUtils.createRealVector(t2);
            if (nPoly > 1) {
                coeff = MatrixUtils.createRealMatrix(nPoly, nc);
                icoeff = MatrixUtils.createRealMatrix(nPoly, nc);
            } else {
                coeff = MatrixUtils.createRealMatrix(nc, 1);
                icoeff = MatrixUtils.createRealMatrix(nc, 1);
            }

            this.xMax = 1;
            this.xMin = 0;
            this.funcName = "legendre";
        }
    }

    public Pair<Integer, RealVector> fit(RealVector xData, RealVector yData, RealVector invVar, RealVector x2) {
        var goodBk = Util.takeIndices(mask, IntStream.range(nOrd, mask.getDimension()).toArray());
        var nn = (int) Util.sum(goodBk);

        if (nn < nOrd) {
            var yFit = MatrixUtils.createRealVector(yData.getDimension());
            return new Pair<>(-1, yFit);
        }

        var nFull = nn * nPoly;
        var bw = nPoly * nOrd;
        var t = action(xData, x2);
        var a1 = t.a();
        var lower = t.b();
        var upper = t.c();
        var foo = MatrixUtils.createRealMatrix(invVar.getDimension(), bw);
        for (int i = 0; i < bw; i++) {
            foo.setColumnVector(i, invVar);//todo check
        }


        var a2 = Util.ebeMultiply(a1, foo);
        var alpha = MatrixUtils.createRealMatrix(bw, (int) (nFull+bw));
        var beta = MatrixUtils.createRealVector((int) (nFull+bw));
        var bi = MatrixUtils.createRealVector(IntStream.range(0, bw).mapToDouble(i -> i).toArray());
        var bo = MatrixUtils.createRealVector(IntStream.range(0, bw).mapToDouble(i -> i).toArray());
        for (int k = 1; k < bw; k++) {
            int finalK = k;
            bi = bi.append(MatrixUtils.createRealVector(IntStream.range(0, bw-k).map(i -> i+(bw+1)* finalK).mapToDouble(i -> i).toArray()));
            bo = bo.append(MatrixUtils.createRealVector(IntStream.range(0, bw-k).map(i -> i+(bw)* finalK).mapToDouble(i -> i).toArray()));
        }

        for (int k = 0; k < nn - nOrd + 1; k++) {
            var iTop = k*nPoly;
            var iBottom = Math.min(iTop, nFull) + bw - 1;
            var ict = upper.getEntry(k) - lower.getEntry(k) + 1;
            if (ict > 0) {
                var work = a1.getSubMatrix(IntStream.range((int) lower.getEntry(k), (int) (upper.getEntry(k)+1)).toArray(), IntStream.range(0, a1.getColumnDimension()).toArray())
                        .transposeMultiply(a2.getSubMatrix(IntStream.range((int) lower.getEntry(k), (int) (upper.getEntry(k)+1)).toArray(), IntStream.range(0, a2.getColumnDimension()).toArray()));
                var wb = a2.getSubMatrix(IntStream.range((int) lower.getEntry(k), (int) (upper.getEntry(k)+1)).toArray(), IntStream.range(0, a2.getColumnDimension()).toArray()).preMultiply(yData.getSubVector((int) lower.getEntry(k), (int) (upper.getEntry(k)+1-lower.getEntry(k))));

                var tm = new Util.TransposeFlatMap(alpha);
                var flatWork = Util.flatten(work, Util.Direction.ROW);
                var t2 = bo.mapAdd(iTop*bw);
                for (int i = 0; i < bi.getDimension(); i++) {
                    tm.set((int) t2.getEntry(i), tm.get((int) t2.getEntry(i)) + flatWork.getEntry((int) bi.getEntry(i)));
                }


                var t1 = IntStream.rangeClosed(iTop, (int) (iBottom+1)).toArray();
                for (int i = 0; i < wb.getDimension(); i++) {
                    beta.addToEntry(t1[i], wb.getEntry(i));
                }
            }
        }
        var minInfluence = 1.0e-10 * Util.sum(invVar) / nFull;
        var errb = BSplineUtil.exofastCholeskyBand(alpha.copy(), minInfluence);//todo broken
        RealMatrix a;
        if (errb.first()[0] == -1) {
            a = errb.second();
        } else {
            var t2 = value(xData, x2, a1, upper, lower);
            return new Pair<>(maskPoints(MatrixUtils.createRealVector(Arrays.stream(errb.first(), 0, errb.first().length).asDoubleStream().toArray())), t2.second());
        }

        var sol = BSplineUtil.exoFastCholeskySolve(a, beta);
        if (nPoly > 1) {
            //todo test
            var aM = Util.reshape(a.getRowVector(0).getSubVector(0, (int) nFull), nPoly, nn);
            var solM = Util.reshape(sol.getSubVector(0, (int) nFull), nPoly, nn);
            var ap = 0;
            var solp = 0;
            for (int i = 0; i < goodBk.getDimension(); i++) {
                if (goodBk.getEntry(i) >= 1) {
                    icoeff.setColumnVector(i, aM.getColumnVector(ap++));
                    coeff.setColumnVector(i, solM.getColumnVector(solp++));
                }
            }
        } else {
            // tested
            var aM = a.getRowVector(0).getSubVector(0, (int) nFull);
            var solM = sol.getSubVector(0, (int) nFull);
            var ap = 0;
            var solp = 0;
            for (int i = 0; i < goodBk.getDimension(); i++) {
                if (goodBk.getEntry(i) >= 1) {
                    icoeff.setEntry(i, 0, aM.getEntry(ap++));
                    coeff.setEntry(i, 0, solM.getEntry(solp++));
                }
            }
        }

        var t2 = value(xData, x2, a1, upper, lower);
        return new Pair<>(0, t2.first());
    }

    public Pair<RealVector, RealVector> value(RealVector x, RealVector x2, RealMatrix action, RealVector upper, RealVector lower) {
        var xSort = IndexedSorting.sort(x);
        var xWork = Util.takeIndices(x, xSort);
        var x2Work = x2 != null ? Util.takeIndices(x2, xSort) : null;
        if (action != null) {
            if (lower == null || upper == null) {
                throw new IllegalArgumentException("Must specify lower and upper if action is set");
            }
        } else {
            var t = action(xWork, x2Work);
            action = t.a();
            lower = t.b();
            upper = t.c();
        }
        var yFit = MatrixUtils.createRealVector(x.getDimension());
        var bw = nPoly * nOrd;
        var goodBk = Util.nonZero(mask);
        var coeffBk = Util.nonZero(mask.getSubVector(nOrd, mask.getDimension()-nOrd));//todo check n size
        var n = (int) Util.sum(mask) - nOrd;
        var goodCoeff = nPoly > 1 ? coeff.getSubMatrix(IntStream.range(0, coeff.getRowDimension()).toArray(), coeffBk) :
                coeff.getSubMatrix(coeffBk, IntStream.range(0, coeff.getColumnDimension()).toArray());

        for (int i = 0; i < n - nOrd + 1; i++) {
            var ict = upper.getEntry(i) - lower.getEntry(i) + 1;
            if (ict > 0) {
                var spot = IntStream.range(0, bw);
                int finalI = i;
                var t = action.getSubMatrix(IntStream.range((int) lower.getEntry(i), (int) (upper.getEntry(i)+1)).toArray(),
                        IntStream.range(0, action.getColumnDimension()).toArray()).multiply(goodCoeff.getSubMatrix(spot.map(e -> finalI *nPoly+e).toArray(), IntStream.range(0, goodCoeff.getColumnDimension()).toArray()));
                yFit.setSubVector(((int) lower.getEntry(i)), t.getColumnVector(0));
            }
        }

        var yy = yFit.copy();
        var p = 0;
        for (int i : xSort) {
            yy.setEntry(i, yFit.getEntry(p++));
        }

        var mask = MatrixUtils.createRealVector(x.getDimension());
        mask.set(1);
        var gb = Util.takeIndices(breakpoints, goodBk);
        //todo this needs testing
        var outside = Util.predicateIndices(x, d -> ((d < gb.getEntry(nOrd-1)) | (d > gb.getEntry(n))));
        if (outside.length > 0) {
            for (int i : outside) {
                mask.setEntry(i, 0);
            }
        }

        var hmm = Util.nonZero(IntStream.range(0, goodBk.length - 1).map(i -> goodBk[i+1] - goodBk[i]).map(i -> i > 2 ? 1 : 0).toArray());
        for (int jj = 0; jj < hmm.length; jj++) {
            //todo untested
            int finalJj = jj;
            var inside = Util.predicateIndices(x, d -> (d >= breakpoints.getEntry(goodBk[hmm[finalJj]]) & d <= breakpoints.getEntry(goodBk[hmm[finalJj]+1]-1)));
            if (inside.length > 0) {
                for (int i : inside) {
                    mask.setEntry(i, 0);
                }
            }
        }

        return new Pair<>(yy, mask);
    }

    public Triple<RealMatrix, RealVector, RealVector> action(RealVector x, RealVector x2) {
        var nx = x.getDimension();
        var nBkpt = (int) Util.sum(mask);

        if (nBkpt < 2*nOrd) {
            return new Triple<>(MatrixUtils.createRealMatrix(new double[][]{new double[]{-2}}), MatrixUtils.createRealVector(1), MatrixUtils.createRealVector(1));
        }

        var n = nBkpt - nOrd;
        var gb = Util.takeMask(breakpoints, mask);
        var bw = nPoly * nOrd;
        var lower = MatrixUtils.createRealVector(n - nOrd + 1);
        var upper = MatrixUtils.createRealVector(n - nOrd + 1).mapAddToSelf(-1);
        var indx = intrv(x);
        var bf1 = bsplvn(x, indx);
        var action = bf1.copy();
        var aa = Util.uniq(indx, IntStream.range(0, indx.getDimension()).toArray());
        for (int i : aa) {
            upper.setEntry((int) (indx.getEntry(i)-nOrd+1), i);
        }

        var rIndx = Util.reverse(indx);
        var bb = Util.uniq(rIndx, IntStream.range(0, rIndx.getDimension()).toArray());
        for (int i : bb) {
            lower.setEntry((int) (rIndx.getEntry(i)-nOrd+1), nx-i-1);
        }
        if (x2 != null) {
            throw new IllegalStateException("not impl.");
            //todo ln241
        }

        return new Triple<>(action, lower, upper);
    }

    public int maskPoints(RealVector err) {
        var nBkpt = (int) Util.sum(mask);
        if (nBkpt <= 2*nOrd) {
            return -2;
        }

        var hmm = Util.takeIndices(err, Util.uniq(err.mapDivide(nPoly), null)).mapDivideToSelf(nPoly);
        var n = nBkpt - nOrd;
        if (Util.isAny(hmm, d -> d >=n)) {
            return -2;
        }

        var test = MatrixUtils.createRealVector(nBkpt);
        for (int jj = (int) -Math.ceil(nOrd / 2D); jj < nOrd / 2D; jj++) {
            var hj = hmm.mapAdd(jj);
            var foo = Util.where(d -> d > 0, hj, hj, MatrixUtils.createRealVector(hmm.getDimension()));
            var fn = foo.mapAdd(nOrd);
            var inside = Util.where(d -> d < n-1, fn, fn, MatrixUtils.createRealVector(hmm.getDimension()).mapAddToSelf(n-1));
            test.walkInOptimizedOrder(new DefaultVectorChangingVisitor() {
                @Override
                public double visit(int index, double value) {
                    if (inside.getEntry(index) > 0) {
                        return 1;
                    }
                    return super.visit(index, value);
                }
            });
        }

        if (Util.isAny(test, d -> d > 0)) {
            var reality = Util.takeMask(mask, test);
            if (Util.isAny(Util.takeMask(mask, reality), d -> d > 0)) {
                mask.walkInOptimizedOrder(new DefaultVectorChangingVisitor() {
                    @Override
                    public double visit(int index, double value) {
                        if (reality.getEntry(index) > 0) {
                            return 0;
                        }
                        return super.visit(index, value);
                    }
                });

                return -1;
            } else {
                return -2;
            }
        } else {
            return -2;
        }
    }

    public RealVector intrv(RealVector x) {
        var gb = Util.takeMask(breakpoints, mask);
        var n = gb.getDimension() - nOrd;
        var indx =MatrixUtils.createRealVector(x.getDimension());
        var iLeft = nOrd - 1;
        for (int i = 0; i < x.getDimension(); i++) {
            while (x.getEntry(i) > gb.getEntry(iLeft+1) && iLeft < n-1) {
                iLeft++;
            }
            indx.setEntry(i, iLeft);
        }

        return indx;
    }

    public RealMatrix bsplvn(RealVector x, RealVector iLeft) {
        var bkpt = Util.takeMask(breakpoints, mask);
        var vnikx = MatrixUtils.createRealMatrix(x.getDimension(), nOrd);
        var deltap = vnikx.copy();
        var deltam = vnikx.copy();

        var j = 0;
        var t = new double[vnikx.getRowDimension()];
        Arrays.fill(t, 1);
        vnikx.setColumnVector(0, MatrixUtils.createRealVector(t));

        while (j < nOrd - 1) {
            var ipj = iLeft.mapAdd(j+1);
            deltap.setColumnVector(j, Util.takeIndices(bkpt, ipj).add(x.mapMultiply(-1)));
            var imj = iLeft.mapAdd(-j);
            deltam.setColumnVector(j, x.subtract(Util.takeIndices(bkpt, imj)));
            RealVector vmPrev = MatrixUtils.createRealVector(deltam.getRowDimension());
            for (int l = 0; l < j+1; l++) {
                var vm = vnikx.getColumnVector(l).ebeDivide(deltap.getColumnVector(l).add(deltam.getColumnVector(j-l)));
                vnikx.setColumnVector(l, deltap.getColumnVector(l).ebeMultiply(vm).add(vmPrev));
                vmPrev = vm.ebeMultiply(deltam.getColumnVector(j-l));
            }

            j++;
            vnikx.setColumnVector(j, vmPrev);
        }

        return vnikx;
    }

    public static Pair<BSpline, RealVector> iterFit(RealVector xData, RealVector yData, BSplineArgs kArgs) {
        RealVector invVar = null;
        double upper = 5;
        double lower = 5;
        RealVector x2 = null;
        int maxIter = 10;

        var nx = xData.getDimension();
        if (yData.getDimension() != nx) {
            throw new IllegalArgumentException("Dimensions of xdata and ydata do not agree.");
        }
        if (invVar != null) {
            if (invVar.getDimension() != nx) {
                throw new IllegalArgumentException("Dimensions of xdata and invvar do not agree.");
            }
        } else {
            var variance = Util.var(yData);
            if (variance == 0) {
                variance = 1;
            }

            var t = new double[yData.getDimension()];
            Arrays.fill(t, 1/variance);
            invVar = MatrixUtils.createRealVector(t);
        }

        if (x2 != null) {
            if (x2.getDimension() != nx) {
                throw new IllegalArgumentException("Dimensions of xdata and x2 do not agree.");
            }
        }

        var yFit = MatrixUtils.createRealVector(nx);
        var outMask = MatrixUtils.createRealVector(nx);
        outMask.set(1);

        var xSort = IndexedSorting.sort(xData);
        var maskWork = MatrixUtils.createRealVector(xSort.length);
        RealVector finalInvVar = invVar;
        maskWork.walkInOptimizedOrder(new RealVectorChangingVisitor() {
            @Override
            public void start(int dimension, int start, int end) {
            }

            @Override
            public double visit(int index, double value) {
                var i = xSort[index];
                return (int) outMask.getEntry(i) & Util.bool2int(finalInvVar.getEntry(i) > 0);
            }

            @Override
            public double end() {
                return 0;
            }
        });

        BSpline sSet;
        if (kArgs.oldSet != null) {
            sSet = kArgs.oldSet;
            sSet.mask.walkInOptimizedOrder(new RealVectorChangingVisitor() {
                @Override
                public void start(int dimension, int start, int end) {

                }

                @Override
                public double visit(int index, double value) {
                    return 1;
                }

                @Override
                public double end() {
                    return 0;
                }
            });
            sSet.coeff.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return 0;
                }
            });
        } else {
            if (!Util.isAny(maskWork, d -> d >= 1)) {
                throw new IllegalStateException("No valid data points.");
            }
            if (kArgs.fullBkpt != null) {
                //todo fullBkpt not used in python?
                throw new IllegalStateException("not impl.");
            } else {
                sSet = new BSplineBuilder().setX(Util.takeIndices(xData, Util.filterIndices(xSort, maskWork))).setkArgs(kArgs).createBSpline();
                if (Util.sum(maskWork) < sSet.nOrd) {
                    System.out.println("Number of good data points fewer than nOrd");
                    return new Pair<>(sSet, outMask);
                }
                if (x2 != null) {
                    var xMin = kArgs.xMin == null ? Util.min(x2) : kArgs.xMin;
                    var xMax = kArgs.xMax == null ? Util.max(x2) : kArgs.xMax;
                    if (xMax == xMin) {
                        xMax = xMin + 1;
                    }
                    sSet.xMax = xMax;
                    sSet.xMin = xMin;
                    if (kArgs.funcName != null) {
                        sSet.funcName = kArgs.funcName;
                    }
                }
            }
        }

        var xWork = Util.takeIndices(xData, xSort);
        var yWork = Util.takeIndices(yData, xSort);
        var invWork = Util.takeIndices(invVar, xSort);
        RealVector x2Work = null;
        if (x2 != null) {
            xWork = Util.takeIndices(x2, xSort);
        }

        var iIter = 0;
        var error = 0;
        var qDone = -1;

        while ((error != 0 || qDone == -1) && iIter <= maxIter) {
            var goodBk = Util.nonZero(sSet.mask);
            if (Util.sum(maskWork) <= 1 || !Util.isAny(sSet.mask, d -> d >= 1)) {
                sSet.coeff.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
                    @Override
                    public double visit(int row, int column, double value) {
                        return 0;
                    }
                });
                iIter = maxIter + 1;
            } else {
                if (kArgs.requireN != null) {
                    var i = 0;
                    while (xWork.getEntry(i) < sSet.breakpoints.getEntry(goodBk[sSet.nOrd]) && i < nx-1) {
                        i++;
                    }
                    var ct = 0;
                    for (int iLeft = sSet.nOrd; iLeft < Util.sum(sSet.mask) - sSet.nOrd + 1; iLeft++) {
                        while (xWork.getEntry(i) >= sSet.breakpoints.getEntry(goodBk[iLeft]) &&
                                xWork.getEntry(i) < sSet.breakpoints.getEntry(goodBk[iLeft+1]) &&
                                i < nx-1) {
                            ct += Util.bool2int(invWork.getEntry(i) * maskWork.getEntry(i) > 0);
                            i++;
                        }

                        if (ct >= kArgs.requireN) {
                            ct = 0;
                        } else {
                            sSet.mask.setEntry(goodBk[iLeft], 0);
                        }
                    }
                }

                var ey = sSet.fit(xWork, yWork, invWork.ebeMultiply(maskWork), x2Work);
                error = ey.first();
                yFit = ey.second();
            }

            iIter++;
            var inMask = maskWork.copy();
            if (error == -2) {
                return new Pair<>(sSet, outMask);
            } else if (error == 0) {
                var mq = BSplineUtil.djsReject(Util.vector2Matrix(yWork), Util.vector2Matrix(yFit),
                        Util.vector2Matrix(maskWork), Util.vector2Matrix(inMask), Util.vector2Matrix(invWork), lower, upper);
                maskWork = Util.flatten(mq.first(), Util.Direction.ROW);
                qDone = Util.bool2int(mq.second());
            }
        }

        var temp = yFit.copy();
        var p = 0;
        for (int i : xSort) {
            outMask.setEntry(i, maskWork.getEntry(p));
            yFit.setEntry(i, temp.getEntry(p++));
        }

        return new Pair<>(sSet, outMask);
    }

    public record BSplineArgs(BSpline oldSet, Object fullBkpt, Double bkSpace, Double xMin, Double xMax, String funcName, Integer requireN) {

    }
}
