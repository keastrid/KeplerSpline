package com.astroimagej.bspline.util;

import org.hipparchus.linear.RealVectorChangingVisitor;

public class DefaultVectorChangingVisitor implements RealVectorChangingVisitor {
    @Override
    public void start(int dimension, int start, int end) {

    }

    @Override
    public double visit(int index, double value) {
        return value;
    }

    @Override
    public double end() {
        return 0;
    }
}
