package org.hyperonline.visiontest2019.runner;

import edu.wpi.first.networktables.NetworkTableInstance;

public class Runner {

    public static void main(String[] args) {
        NetworkTableInstance inst = NetworkTableInstance.getDefault();
        inst.setNetworkIdentity("Robot");
        inst.startServer(System.getProperty("user.home") + "/networktables.ini");
        @SuppressWarnings("unused")
        VisionSystem vs = new VisionSystem();
        while(true) { 
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
            }
        }
    }

}
