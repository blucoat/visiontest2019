package org.hyperonline.visiontest2019.pipelines;

import org.hyperonline.hyperlib.vision.VisionResult;

public class SkewVisionResult extends VisionResult {

    private final double m_skew;
    
    public SkewVisionResult(double xError, double yError, double xAbs, double yAbs, double skew, boolean foundTarget) {
        super(xError, yError, xAbs, yAbs, foundTarget);
        m_skew = skew;
    }
    
    public double skew() {
        return m_skew;
    }

}
