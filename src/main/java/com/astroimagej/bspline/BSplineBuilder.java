package com.astroimagej.bspline;

import org.hipparchus.linear.RealVector;

class BSplineBuilder {
    private RealVector x;
    private int nOrd = 4;
    private int nPoly = 1;
    private RealVector bkpt = null;
    private double bkSpread = 1D;
    private BSpline.BSplineArgs kArgs = null;

    public BSplineBuilder setX(RealVector x) {
        this.x = x;
        return this;
    }

    public BSplineBuilder setnOrd(int nOrd) {
        this.nOrd = nOrd;
        return this;
    }

    public BSplineBuilder setnPoly(int nPoly) {
        this.nPoly = nPoly;
        return this;
    }

    public BSplineBuilder setBkpt(RealVector bkpt) {
        this.bkpt = bkpt;
        return this;
    }

    public BSplineBuilder setBkSpread(double bkSpread) {
        this.bkSpread = bkSpread;
        return this;
    }

    public BSplineBuilder setkArgs(BSpline.BSplineArgs kArgs) {
        this.kArgs = kArgs;
        return this;
    }

    public BSpline createBSpline() {
        return new BSpline(x, nOrd, nPoly, bkpt, bkSpread, kArgs);
    }
}