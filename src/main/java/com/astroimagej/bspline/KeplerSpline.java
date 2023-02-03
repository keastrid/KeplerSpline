package com.astroimagej.bspline;

import com.astroimagej.bspline.util.Pair;
import com.astroimagej.bspline.util.Triple;
import com.astroimagej.bspline.util.Util;
import org.hipparchus.linear.MatrixUtils;
import org.hipparchus.linear.RealVector;

import java.util.*;

public class KeplerSpline {
    /**
     * In python, this handles sets of vectors. Here all vectors must be combined first
     */
    private static Pair<LinkedList<RealVector>, LinkedList<RealVector>> split(RealVector allTime, RealVector allFlux, Double gapWidth) {
        if (!Util.shapeMatches(allFlux, allTime)) {
            throw new IllegalArgumentException("Shapes must match.");
        }

        if (gapWidth == null) {
            gapWidth = 0.75;
        }

        var outTime = new LinkedList<RealVector>();
        var outFlux = new LinkedList<RealVector>();

        var start = 0;
        for (int end = 1; end < allTime.getDimension() + 1; end++) {
            if (end == allTime.getDimension() || allTime.getEntry(end) - allTime.getEntry(end-1) > gapWidth) {
                outTime.add(allTime.getSubVector(start, end-start));
                outFlux.add(allFlux.getSubVector(start, end-start));
                start = end;
            }
        }

        if (outTime.size() != outFlux.size()) {
            throw new IllegalStateException("Failed to split correctly");
        }

        return new Pair<>(outTime, outFlux);
    }

    private static Triple<Double, Double, RealVector> robustMean(RealVector y, double cut) {
        var median = Util.median(y);
        var absDev = MatrixUtils.createRealVector(Arrays.stream(y.toArray()).map(d -> Math.abs(d - median)).toArray());
        var sigma = 1.4826 * Util.median(absDev);

        if (sigma < 1.0e-24) {
            sigma = 1.253 * Util.mean(absDev.toArray());
        }

        double finalSigma = sigma;
        var mask = absDev.map(d -> Util.bool2int(d <= cut * finalSigma));

        sigma = Util.std(Util.takeMask(y, mask));

        var sc = Math.max(cut, 1);
        if (sc <= 4.5) {
            sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * (sc * sc) + 0.020142 * sc * sc * sc);
        }

        double finalSigma1 = sigma;
        mask = absDev.map(d -> Util.bool2int(d <= cut * finalSigma1));

        sigma = Util.std(Util.takeIndices(y, mask));
        sc = Math.max(cut, 1);

        if (sc <= 4.5) {
            sigma /= (-0.15405 + 0.90723 * sc - 0.23584 * (sc * sc) + 0.020142 * sc * sc * sc);
        }

        var mean = Util.mean(Util.takeIndices(y, mask).toArray());
        var meanStdDev = sigma / Math.sqrt(y.getDimension() - 1);

        return new Triple<>(mean, meanStdDev, mask);
    }

    private static Pair<RealVector, RealVector> keplerSpline(RealVector time, RealVector flux, Double bkSpace, Integer maxIter,
                                                            Double outlierCut, RealVector inputMask) {
        if (bkSpace == null) {
            bkSpace = 1.5;
        }

        if (maxIter == null) {
            maxIter = 5;
        }

        if (outlierCut == null) {
            outlierCut = 3D;
        }

        if (time.getDimension() < 4) {
            throw new InsufficientPointsError("Cannot fit curve of 4 or less points");
        }

        // Rescale time to [0, 1]
        var stats = Arrays.stream(time.toArray()).summaryStatistics();
        var tMin = stats.getMin();
        var tMax = stats.getMax();
        time.mapToSelf(t -> (t - tMin)/(tMax-tMin));
        bkSpace /= (tMax - tMin); // Rescale bucket spacing

        // Values of the best fitting spline evaluated at the time points
        RealVector spline = null;

        // Mask indicating the points used to fit the spline
        RealVector mask = null;

        if (inputMask == null) {
            inputMask = MatrixUtils.createRealVector(time.getDimension());
            inputMask.set(1);
        }

        for (int i = 0; i < maxIter; i++) {
            if (spline == null) {
                mask = inputMask; // Try to fit all points in input mask
            } else {
                var residuals = flux.subtract(spline);
                var newMask = robustMean(residuals, outlierCut).c();
                newMask = Util.and(newMask, inputMask);
                if (newMask.equals(mask)) {
                    break;
                }

                mask = newMask;
            }

            if (Util.sum(mask) < 4) {
                throw new InsufficientPointsError("Ran out of points");
            }

            try {
                var curve = BSpline.iterFit(Util.takeMask(time, mask),
                        Util.takeMask(flux, mask),
                        new BSpline.BSplineArgs(null, null, bkSpace,
                                null, null, null, null)).first();

                spline = curve.value(time, null, null, null, null).first();
            } catch (Exception e) {
                throw new SplineError(e);
            }
        }

        return new Pair<>(spline, mask);
    }

    private static Pair<Collection<RealVector>, SplineMetadata> chooseKeplerSpline(Collection<RealVector> allTime, Collection<RealVector> allFlux,
                                           double[] bkSpaces, Integer maxIter, Double outlierCut, Double penaltyCoeff,
                                           boolean verbose, Collection<RealVector> allInputMask) {
        if (maxIter == null) {
            maxIter = 5;
        }

        if (penaltyCoeff == null) {
            penaltyCoeff = 1D;
        }

        Collection<RealVector> bestSpline = null;
        var metadata = new SplineMetadata();

        var scaledDiffs1 = allFlux.stream().map(Util::diff).map(v -> v.mapDivideToSelf(Math.sqrt(2))).toList();
        var scaledDiffs = Util.concatenate(scaledDiffs1);
        if (scaledDiffs.getDimension() < 1) {
            bestSpline = allFlux.stream().map(v -> {
                var v0 = MatrixUtils.createRealVector(v.getDimension());
                v0.set(Double.NaN);
                return v0;
            }).toList();
            metadata.lightCurveMask = allFlux.stream().map(v -> MatrixUtils.createRealVector(v.getDimension())).toList();
            return new Pair<>(bestSpline, metadata);
        }

        var sigma = Util.median(scaledDiffs.map(Math::abs)) * 1.48;

        if (allInputMask == null || allInputMask.isEmpty() || Util.isAll(allInputMask, Objects::isNull)) {
            allInputMask = allFlux.stream().map(v -> {
                var v0 = MatrixUtils.createRealVector(v.getDimension());
                v0.set(1);
                return v0;
            }).toList();
        }

        var allTime2 = new ArrayList<>(allTime);
        var allFlux2 = new ArrayList<>(allFlux);
        var allInputMask2 = new ArrayList<>(allInputMask);
        for (double bkSpace : bkSpaces) {
            var nParams = 0;
            var nPoints= 0;
            var ssr = 0D;

            var spline = new LinkedList<RealVector>();
            var lightCurveMask = new LinkedList<RealVector>();
            var badBkSpace = false;

            RealVector splinePiece;
            RealVector mask;
            for (int i = 0; i < allTime.size(); i++) {
                var time = allTime2.get(i);
                var flux = allFlux2.get(i);
                var thisInputMask = allInputMask2.get(i);

                try {
                    var t = keplerSpline(time.copy(), flux.copy(), bkSpace, maxIter, outlierCut, thisInputMask.copy());
                    splinePiece = t.first();
                    mask = t.second();
                } catch (InsufficientPointsError e) {
                    if (verbose) {
                        e.printStackTrace();
                    }
                    var v = MatrixUtils.createRealVector(flux.getDimension());
                    v.set(Double.NaN);
                    spline.add(v);
                    lightCurveMask.add(MatrixUtils.createRealVector(flux.getDimension()));
                    continue;
                } catch (SplineError e) {
                    if (verbose) {
                        e.printStackTrace();
                    }

                    metadata.badBkspaces.add(bkSpace);
                    badBkSpace = true;
                    continue;
                }

                spline.add(splinePiece);
                lightCurveMask.add(mask);

                var t1 = Arrays.stream(time.toArray()).summaryStatistics();
                var totalTime = t1.getMax() - t1.getMin();
                var nKnots = (int) (totalTime/bkSpace + 1);
                nParams += nKnots + 3 - 1;
                nPoints += Util.sum(mask);
                ssr += Util.sum(Util.takeMask(flux, mask).subtract(Util.takeMask(splinePiece, mask)).mapToSelf(d -> d*d));
            }

            if (badBkSpace || nPoints == 0) {
                continue;
            }

            var likelihoodTerm = nPoints * Math.log(2 * Math.PI * sigma * sigma) + ssr / (sigma * sigma);
            var penaltyTerm = nParams * Math.log(nPoints);

            var bic = likelihoodTerm + penaltyCoeff * penaltyTerm;

            if (bestSpline == null || bestSpline.isEmpty() || metadata.bic == null || bic < metadata.bic) {
                bestSpline = spline;
                metadata.lightCurveMask = lightCurveMask;
                metadata.inputLightCurveMask = allInputMask;
                metadata.bkSpace = bkSpace;
                metadata.likelihoodTerm = likelihoodTerm;
                metadata.penaltyTerm = penaltyTerm;
                metadata.bic = bic;
            }
        }

        if (bestSpline == null || bestSpline.isEmpty()) {
            bestSpline = allFlux.stream().map(v -> {
                var v0 = MatrixUtils.createRealVector(v.getDimension());
                v0.set(Double.NaN);
                return v0;
            }).toList();
            metadata.lightCurveMask = allFlux.stream().map(v -> MatrixUtils.createRealVector(v.getDimension())).toList();
            metadata.inputLightCurveMask = allFlux.stream().map(v -> MatrixUtils.createRealVector(v.getDimension())).toList();
        }

        return new Pair<>(bestSpline, metadata);
    }

    private static Pair<Collection<RealVector>, SplineMetadata> fitKeplerSpline(Collection<RealVector> allTime, Collection<RealVector> allFlux, Double bkSpaceMin,
                                                                               Double bkSpaceMax, Integer bkSpaceNum, Integer maxIter, Double penaltyCoeff, boolean verbose) {
        if (bkSpaceNum == null) {
            bkSpaceNum = 20;
        }

        if (bkSpaceMax == null) {
            bkSpaceMax = 20D;
        }

        if (bkSpaceMin == null) {
            bkSpaceMin = 0.5;
        }

        if (maxIter == null) {
            maxIter = 5;
        }

        if (penaltyCoeff == null) {
            penaltyCoeff = 1D;
        }

        var bkSpaces= Util.logSpace(Math.log10(bkSpaceMin), Math.log10(bkSpaceMax), bkSpaceNum);
        return chooseKeplerSpline(allTime, allFlux, bkSpaces, maxIter, penaltyCoeff, null, verbose, null);
    }

    public static Pair<RealVector, SplineMetadata> chooseKeplerSplineV2(RealVector time, RealVector flux) {
        return chooseKeplerSplineV2(time, flux, null, null, null, null, null, null, null, false);
    }

    public static Pair<RealVector, SplineMetadata> chooseKeplerSplineV2(RealVector time, RealVector flux, Double bkSpaceMin, Double bkSpaceMax,
                                                                        Integer bkSpaceNum, RealVector inputMask, Double gapWidthIn,
                                                                        Integer maxIter, Double outlierCut, boolean returnMetadata) {
        if (bkSpaceNum == null) {
            bkSpaceNum = 20;
        }

        if (bkSpaceMax == null) {
            bkSpaceMax = 20D;
        }

        if (bkSpaceMin == null) {
            bkSpaceMin = 0.5;
        }

        if (gapWidthIn == null) {
            gapWidthIn = bkSpaceMin;
        }

        if (inputMask == null) {
            inputMask = MatrixUtils.createRealVector(time.getDimension());
            inputMask.set(1);
        }

        var t = split(time, flux, gapWidthIn);
        var allTime = t.first();
        var allFlux = t.second();
        t = split(time, inputMask, gapWidthIn);
        var allTime2 = t.first();
        var allInputMask = t.second();

        var bkSpaces = Util.logSpace(Math.log10(bkSpaceMin), Math.log10(bkSpaceMax), bkSpaceNum);

        var t1 = chooseKeplerSpline(allTime, allFlux, bkSpaces, maxIter, outlierCut, null, false, allInputMask);
        var splines = t1.first();
        var metadata = t1.second();

        var spline = Util.concatenate(splines);

        //todo how to handle - python merges the list into one array for metadata
        return new Pair<>(spline, metadata);
    }

    public static Pair<RealVector, SplineMetadata> keplerSplineV2(RealVector time, RealVector flux) {
        return keplerSplineV2(time, flux, null, null, null, null, null, false);
    }

    public static Pair<RealVector, SplineMetadata> keplerSplineV2(RealVector time, RealVector flux, Double bkSpace, RealVector inputMask,
                                                                  Double gapWidthIn, Integer maxIter, Double outlierCut,
                                                                  boolean returnMetadata) {
        if (bkSpace == null) {
            bkSpace = 1.5;
        }

        if (gapWidthIn == null) {
            gapWidthIn = bkSpace;
        }

        if (inputMask == null) {
            inputMask = MatrixUtils.createRealVector(time.getDimension());
            inputMask.set(1);
        }

        var t = split(time, flux, gapWidthIn);
        var allTime = t.first();
        var allFlux = t.second();
        t = split(time, inputMask, gapWidthIn);
        var allTime2 = t.first();
        var allInputMask = t.second();

        var t1 = chooseKeplerSpline(allTime, allFlux, new double[]{bkSpace}, maxIter, outlierCut, null, false, allInputMask);
        var splines = t1.first();
        var metadata = t1.second();

        var spline = Util.concatenate(splines);

        //todo how to handle - python merges the list into one array for metadata
        return new Pair<>(spline, metadata);
    }


    public static class SplineMetadata {
        public Collection<RealVector> lightCurveMask = null;
        public Collection<RealVector> inputLightCurveMask = null;
        public Double bkSpace = null;
        public Collection<Double> badBkspaces = new ArrayList<>();
        public Double likelihoodTerm = null;
        public Double penaltyTerm = null;
        public Double bic = null;
    }

    private static class InsufficientPointsError extends IllegalArgumentException {
        public InsufficientPointsError(String s) {
            super(s);
        }
    }
    private static class SplineError extends IllegalArgumentException {
        public SplineError(Exception e) {
            super(e);
        }
    }
}
