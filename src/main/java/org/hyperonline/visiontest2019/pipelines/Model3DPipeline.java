package org.hyperonline.visiontest2019.pipelines;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import org.hyperonline.hyperlib.pref.DoublePreference;
import org.hyperonline.hyperlib.pref.PreferencesSet;
import org.hyperonline.hyperlib.pref.ScalarPreference;
import org.hyperonline.hyperlib.vision.VisionGUIPipeline;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import edu.wpi.cscore.CvSource;
import edu.wpi.first.cameraserver.CameraServer;

public class Model3DPipeline implements VisionGUIPipeline {
    private final String m_name;

    /*
     * Preferences
     */
    private final PreferencesSet m_prefs;
    private final ScalarPreference m_lowerBound;
    private final ScalarPreference m_upperBound;
    private final DoublePreference m_minArea;
    private final DoublePreference m_focalLength;

    /**
     * Construct a new pipeline with the given name and target processor.
     * 
     * @param name
     *                      The name used to define the preferences set associated
     *                      to this pipeline.
     * @param processor
     *                      Interface specifying how to extract the result from a
     *                      list of rectangles.
     */
    public Model3DPipeline(String name) {
        m_name = Objects.requireNonNull(name);

        m_prefs = new PreferencesSet(name);
        m_lowerBound = m_prefs.addScalar("LowerBound", "HSV", 30, 200, 100);
        m_upperBound = m_prefs.addScalar("UpperBound", "HSV", 80, 255, 255);
        m_minArea = m_prefs.addDouble("MinArea", 20);
        m_focalLength = m_prefs.addDouble("FocalLength", 100);
    }

    /*
     * Intermediate steps used in processing. There's no need to store them between
     * frames, but we do so just to avoid reallocating them each time.
     */
    private final Mat m_hsv = new Mat();
    private final Mat m_filtered = new Mat();

    /**
     * Intermediate result of pairing up two rotated rectangles
     */
    private static class RectPair {
        public RotatedRect left;
        public RotatedRect right;

        public RectPair(RotatedRect left, RotatedRect right) {
            this.left = left;
            this.right = right;
        }

        public Point[] corners() {
            Point[] left_arr = new Point[4];
            Point[] right_arr = new Point[4];
            Point[] pts = new Point[8];
            left.points(left_arr);
            right.points(right_arr);
            Arrays.sort(left_arr, Comparator.comparingDouble(p -> p.y));
            Arrays.sort(right_arr, Comparator.comparingDouble(p -> p.y));
            for (int i = 0; i < 4; i++) {
                pts[2 * i] = left_arr[i];
                pts[2 * i + 1] = right_arr[i];
            }
            return pts;
        }
    }

    /*
     * Debug camera source to make tweaking preferences easier This isn't
     * initialized until the first time process is called. That way I can set the
     * dimensions to be the same as the incoming image automatically.
     */
    private CvSource m_debugSource = null;
    private CvSource m_overheadSource = null;

    private void putFilteredImage(Mat mat) {
        if (m_debugSource == null) {
            m_debugSource = CameraServer.getInstance().putVideo(m_name + " debug stream (filter)", mat.width(),
                    mat.height());
        }
        m_debugSource.putFrame(mat);
    }

    private void putOverheadImage(Mat mat) {
        if (m_overheadSource == null) {
            m_overheadSource = CameraServer.getInstance().putVideo(m_name + " debug stream (overhead)", mat.width(),
                    mat.height());
        }
        m_overheadSource.putFrame(mat);
    }

    private volatile List<Model3DResult> m_lastResult = Collections.emptyList();
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void process(Mat mat) {
        List<MatOfPoint> contours = findTargetContours(mat);
        List<RectPair> pairs = filterAndGroupTargets(contours);
        contours.forEach(MatOfPoint::release);
        m_lastResult = pairs.stream()
                .map(RectPair::corners)
                .map(this::imagePointsToResult)
                .sorted(Comparator.comparingDouble(r -> -Math.abs(r.getX())))
                .collect(Collectors.toUnmodifiableList());
    }

    private List<MatOfPoint> findTargetContours(Mat mat) {
        Imgproc.cvtColor(mat, m_hsv, Imgproc.COLOR_BGR2HSV);
        Core.inRange(m_hsv, m_lowerBound.get(), m_upperBound.get(), m_filtered);
        Imgproc.erode(m_filtered, m_filtered, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5)));
        Imgproc.dilate(m_filtered, m_filtered, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5)));
        putFilteredImage(m_filtered);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat heirarchy = new Mat();
        Imgproc.findContours(m_filtered, contours, heirarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        heirarchy.release();
        return contours;
    }

    private List<RectPair> filterAndGroupTargets(List<MatOfPoint> contours) {
        List<RotatedRect> rectangles = contours.stream()
                .map(Model3DPipeline::minAreaRect)
                .filter(r -> r.size.area() >= m_minArea.get())
                .sorted(Comparator.comparingDouble(r -> r.center.x))
                .collect(Collectors.toList());

        List<RectPair> pairs = new ArrayList<>();
        RotatedRect lastLeft = null;
        for (RotatedRect rect : rectangles) {
            if (isLeft(rect)) {
                lastLeft = rect;
            } else if (lastLeft != null) {
                pairs.add(new RectPair(lastLeft, rect));
                lastLeft = null;
            }
        }

        return pairs;
    }

    private static RotatedRect minAreaRect(MatOfPoint contour) {
        MatOfPoint2f cvtd = new MatOfPoint2f(contour.toArray());
        RotatedRect result = Imgproc.minAreaRect(cvtd);
        cvtd.release();
        return result;
    }

    private static boolean isLeft(RotatedRect rect) {
        if (Math.tan(rect.angle * Math.PI / 180) > 0) {
            return rect.size.width <= rect.size.height;
        } else {
            return rect.size.width > rect.size.height;
        }
    }

    private static final double S14_5 = Math.sin(14.5 * Math.PI / 180);
    private static final double C14_5 = Math.cos(14.5 * Math.PI / 180);
    private static final MatOfPoint3f OBJECT_POINTS = new MatOfPoint3f(
            new Point3(-4 - 2 * C14_5, -5 * C14_5 - 2 * S14_5, 0),
            new Point3(4 + 2 * C14_5, -5 * C14_5 - 2 * S14_5, 0),
            new Point3(-4, -5 * C14_5, 0),
            new Point3(4, -5 * C14_5, 0),
            new Point3(-4 - 5 * S14_5 - 2 * C14_5, -2 * S14_5, 0),
            new Point3(4 + 5 * S14_5 + 2 * C14_5, -2 * S14_5, 0),
            new Point3(-4 - 5 * S14_5, 0, 0),
            new Point3(4 + 5 * S14_5, 0, 0));

    private Model3DResult imagePointsToResult(Point[] pts) {
        updateCameraMatrix();
        Mat rvec = new Mat();
        Mat tvec = new Mat();
        MatOfPoint2f imagePoints = new MatOfPoint2f(pts);

        Calib3d.solvePnP(OBJECT_POINTS, imagePoints, m_cameraMatrix, m_distortion, rvec, tvec);
        Model3DResult res = new Model3DResult(tvec, rvec);

        rvec.release();
        tvec.release();
        imagePoints.release();

        return res;
    }

    private final Mat m_cameraMatrix = new Mat(new Size(3, 3), CvType.CV_32F);
    private final MatOfDouble m_distortion = new MatOfDouble();

    private void updateCameraMatrix() {
        float f = (float) m_focalLength.get();
        float[] data = { f, 0, m_hsv.width() / 2.0f, 0, f, m_hsv.height() / 2.0f, 0, 0, 1 };
        m_cameraMatrix.put(0, 0, data);
    }

    /*
     * Constants for drawing indicators. These could be made into preferences, but
     * do we really care that much?
     */
    private static final Scalar QUAD_COLOR = new Scalar(0, 255, 255);
    private static final Scalar CORNER_COLOR = new Scalar(0, 255, 255);

    private static final MatOfPoint3f QUAD_POINTS = new MatOfPoint3f(
            new Point3(-8, -6, 0),
            new Point3(8, -6, 0),
            new Point3(-8, 1, 0),
            new Point3(8, 1, 0));
    
    public List<Model3DResult> getLastResult() {
        return m_lastResult;
    }
    
    private void drawTargetIndicator(Mat mat, Model3DResult result) {
        Mat tvec = new Mat();
        Mat rvec = new Mat();
        MatOfPoint2f imagePoints = new MatOfPoint2f();
        result.rotation(rvec);
        result.translation(tvec);
        
        Calib3d.projectPoints(OBJECT_POINTS, rvec, tvec, m_cameraMatrix, m_distortion, imagePoints);
        for (Point p : imagePoints.toArray()) {
            Imgproc.circle(mat, p, 4, CORNER_COLOR);
        }
        Calib3d.projectPoints(QUAD_POINTS, rvec, tvec, m_cameraMatrix, m_distortion, imagePoints);
        Point[] imgpts = imagePoints.toArray();
        Imgproc.line(mat, imgpts[0], imgpts[1], QUAD_COLOR);
        Imgproc.line(mat, imgpts[1], imgpts[3], QUAD_COLOR);
        Imgproc.line(mat, imgpts[3], imgpts[2], QUAD_COLOR);
        Imgproc.line(mat, imgpts[2], imgpts[0], QUAD_COLOR);
        
        tvec.release();
        rvec.release();
        imagePoints.release();
    }
    
    private static final int PIX_PER_INCH = 2;
    private static final int INCHES_PER_TICK = 20;
    private static final Scalar OVERHEAD_COLOR = new Scalar(255, 255, 255);
    
    private void drawOverheadImage(Mat mat) {
        mat.setTo(new Scalar(0, 0, 0));
        for (int i = -4; i <= 4; i++) {
            int t = i * INCHES_PER_TICK;
            Imgproc.putText(mat, Integer.toString(t), new Point(320, 240 - t * PIX_PER_INCH), 
                    Core.FONT_HERSHEY_PLAIN, 1.0, new Scalar(255, 255, 255));
        }
        for (Model3DResult target : getLastResult()) {
            // TODO: correct for camera tilt
            float x = 320 + target.getX() * PIX_PER_INCH;
            float y = 240 - target.getZ() * PIX_PER_INCH;
            double angle = Math.atan2(-target.getX(), target.getZ()) - target.topDownAngle() * Math.PI / 180;
            
            double s = Math.sin(angle) * 8 * PIX_PER_INCH;
            double c = Math.cos(angle) * 8 * PIX_PER_INCH;
            Imgproc.line(mat, new Point(x + c, y - s), new Point(x - c, y + s), OVERHEAD_COLOR);
        }
    }
    
    private Mat m_overheadImage = new Mat(480, 640, CvType.CV_8UC3);
    
    /**
     * {@inheritDoc}
     */
    @Override
    public void writeOutput(Mat mat) {
        for (Model3DResult target : getLastResult()) {
            drawTargetIndicator(mat, target);
        }
        drawOverheadImage(m_overheadImage);
        putOverheadImage(m_overheadImage);
    }

}
