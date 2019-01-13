package org.hyperonline.visiontest2019.runner;

import org.hyperonline.hyperlib.pref.IntPreference;
import org.hyperonline.hyperlib.pref.PreferencesSet;
import org.hyperonline.hyperlib.pref.StringPreference;
import org.hyperonline.hyperlib.vision.CrosshairsPipeline;
import org.hyperonline.hyperlib.vision.VisionGUIPipeline;
import org.hyperonline.hyperlib.vision.VisionModule;
import org.hyperonline.visiontest2019.pipelines.Model3DPipeline;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import edu.wpi.cscore.CameraServerJNI;
import edu.wpi.cscore.CvSource;
import edu.wpi.first.cameraserver.CameraServer;

public class VisionSystem {
    
    CvSource m_source;
    Mat m_image;
    Thread m_feederThread;
    
    void imageFeederThread() {
        double fps = m_source.getVideoMode().fps;
        long timeStep = (long) (1e9 / fps);
        long lastTime = System.nanoTime();
        long currentTime;
        
        System.out.println("Vision feeder thread started");
        
        while (true) {
            m_source.putFrame(m_image);
            
            // wait a bit
            while ((currentTime = System.nanoTime()) < lastTime + timeStep) {
                try { Thread.sleep(1); } catch (InterruptedException e) { e.printStackTrace(); }
            }
            lastTime = currentTime;
        }
        
    }
    
    private VisionModule m_module;
    private VisionGUIPipeline m_pipeline;
    private CrosshairsPipeline m_crosshairs;
    
    private PreferencesSet m_prefs = new PreferencesSet("Vision");
    private IntPreference m_xCross = m_prefs.addInt("Crosshairs X", 200);
    private IntPreference m_yCross = m_prefs.addInt("Crosshairs Y", 200);
    private StringPreference m_filename = m_prefs.addString("Image Filename", 
            "/home/james/Robotics/2019VisionImages-1/RocketPanelAngleDark60in.jpg");
    

    public VisionSystem() {
        CameraServerJNI.forceLoad();
        System.out.println(Core.getBuildInformation());
        
        // Set up dummy source to feed images
        m_image = Imgcodecs.imread(m_filename.get());
        if (m_image.empty()) {
            System.out.println("Could not load test image!  Vision will not work!");
            return;
        }
        m_source = CameraServer.getInstance().putVideo("Dummy source of file", m_image.width(), m_image.height());
        m_feederThread = new Thread(this::imageFeederThread);
        m_feederThread.setName("Image feeder thread");
        m_feederThread.setDaemon(true);
        m_feederThread.start();
        
        //m_processor = new SkewPairTargetProcessor(m_xCross::get, m_yCross::get);
        //m_pipeline = new FindTargetsPipeline("My Pipeline", m_processor);
        m_pipeline = new Model3DPipeline("Model3D Pipeline");
        m_crosshairs = new CrosshairsPipeline(m_xCross::get, m_yCross::get, 100, 100, 100);
        
        m_module = new VisionModule.Builder(m_source)
                .addPipeline(m_pipeline)
                .addPipeline(m_crosshairs)
                .build();
        
        m_module.start();
    }

}
