package org.hyperonline.visiontest2019.pipelines;

import java.util.Objects;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

/**
 * Holds the position and orientation of a vision target in 3d space.
 * 
 * @author James Hagborg
 */
public class Model3DResult {

    private final float[] m_tvec, m_rvec;

    /**
     * Create a result from the tvec and rvec values given by the OpenCV method
     * solvePnP. These must be 3x1 row vectors holding 32-bit floats.
     * 
     * @param tvec
     *                 The translation vector.
     * @param rvec
     *                 The position vector.
     */
    public Model3DResult(Mat tvec, Mat rvec) {
        Objects.requireNonNull(tvec, "tvec == null");
        Objects.requireNonNull(rvec, "rvec == null");
        if (tvec.height() != 3 || tvec.width() != 1 || tvec.channels() != 1 || tvec.type() != CvType.CV_32F) {
            throw new IllegalArgumentException("tvec is not a 3x1 column vector");
        }
        if (rvec.height() != 3 || rvec.width() != 1 || rvec.channels() != 1 || rvec.type() != CvType.CV_32F) {
            throw new IllegalArgumentException("rvec is not a 3x1 column vector");
        }

        m_tvec = new float[3];
        m_rvec = new float[3];

        tvec.get(0, 0, m_tvec);
        rvec.get(0, 0, m_rvec);
    }

    /**
     * Copy the translation vector to a given Mat.
     * 
     * @param tvec
     *                 Will hold the translation vector.
     */
    public void translation(Mat tvec) {
        tvec.create(new Size(1, 3), CvType.CV_32F);
        tvec.put(0, 0, m_tvec);
    }

    /**
     * Copy the rotation vector to a given Mat.
     * 
     * @param rvec
     *                 Will hold the rotation vector.
     */
    public void rotation(Mat rvec) {
        rvec.create(new Size(1, 3), CvType.CV_32F);
        rvec.put(0, 0, m_rvec);
    }

    /**
     * Get the corresponding 3x3 rotation matrix.
     * 
     * @param rotMat
     *                   Will hold the rotation matrix.
     */
    public void rotationMatrix(Mat rotMat) {
        Mat rvecMat = new Mat(new Size(1, 3), CvType.CV_32F);
        rvecMat.put(0, 0, m_rvec);
        Calib3d.Rodrigues(rvecMat, rotMat);
        rvecMat.release();
    }

    /**
     * Get the x-coordinate of translation in 3d space, in inches. Positive values
     * represent objects to the right of the camera.
     * 
     * @return The x-coordinate.
     */
    public float getX() {
        return m_tvec[0];
    }

    /**
     * Get the y-coordinate of the translation in 3d space, in inches. Positive
     * values represent objects below the camera.
     * 
     * @return The y-coordinate.
     */
    public float getY() {
        return m_tvec[1];
    }

    /**
     * Get the z-coordinate of the translation in 3d space, in inches. Positive
     * values represent objects in front of the camera.
     * 
     * @return
     */
    public float getZ() {
        return m_tvec[2];
    }

    /**
     * Get the angle of the robot from the point of view of the target. Positive
     * values mean that the robot is to the left of the target, and negative values
     * mean the robot is to the right. The angle is measured in degrees.
     * 
     * This value only represents the position of the robot relative to the target,
     * and so should not depend on the robot's orientation. It does depend on the
     * orientation of the target. It should not depend on the height or angle of the
     * camera, only its position.
     * 
     * @return The angle, in degrees.
     */
    public double topDownAngle() {
        Mat rotMat = new Mat();
        rotationMatrix(rotMat);
        Mat tvec = new Mat();
        translation(tvec);
        Mat noMat = new Mat(new Size(1, 3), CvType.CV_32F);
        Mat transInModelCoords = new Mat();

        // Compute -1 * (rotMat)^T * tvec
        Core.gemm(rotMat, tvec, -1.0, noMat, 0.0, transInModelCoords, Core.GEMM_1_T);
        double x = transInModelCoords.get(0, 0)[0];
        double z = transInModelCoords.get(2, 0)[0];
        
        rotMat.release();
        tvec.release();
        noMat.release();
        transInModelCoords.release();

        return Math.atan2(x, -z) * 180 / Math.PI;
    }
}
