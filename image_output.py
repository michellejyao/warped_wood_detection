import os
import sys
import time
import cv2
import numpy as np
import depthai as dai
from constants import (
    RGB_IMAGE_PATH, DEPTH_MAP_PATH, CAMERA_RESOLUTION,
    STEREO_DEPTH_CONFIG
)

# ---- Tunables ----
FALLBACK_BRIGHTNESS_MEAN = 40        # if mean(BGR) < this, push manual exposure
MANUAL_EXPOSURE_US = 22000           # ~1/45s, adjust to your light
MANUAL_ISO = 800                     # bump if still dark (but more noise)
TARGET_FPS = 10                      # lower FPS => longer AE exposures in dim light
STABILIZE_SECS = 2.0                 # let AE/AWB settle
# -------------------

def detect_camera(dai):
    try:
        devices = dai.Device.getAllConnectedDevices()
        if not devices:
            print("No OAK-D devices found")
            return None
        print(f"✓ Found {len(devices)} OAK-D device(s)")
        for i, device in enumerate(devices):
            print(f"  Device {i+1}: {device.getMxId()}")
        return devices[0]
    except Exception as e:
        print(f"✗ Error detecting camera: {e}")
        return None

def create_camera_pipeline(dai):
    try:
        pipeline = dai.Pipeline()

        # --- Color camera (use ISP output for proper tone mapping) ---
        rgb_cam = pipeline.create(dai.node.ColorCamera)
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        rgb_cam.setResolution(getattr(dai.ColorCameraProperties.SensorResolution, CAMERA_RESOLUTION['rgb']))
        rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        rgb_cam.setInterleaved(False)
        rgb_cam.setFps(TARGET_FPS)
        # If you want a specific output size, uncomment this:
        # rgb_cam.setIspScale(1, 1)  # full ISP res from sensor
        # rgb_cam.setVideoSize(1920, 1080)  # or use setPreviewSize

        # 3A (auto-exposure/white-balance) initial controls
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()  # let device drive exposure
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        ctrl.setAntiBandingMode(dai.CameraControl.AntiBandingMode.AUTO)
        # Small positive EV to brighten a bit
        ctrl.setAutoExposureCompensation(+2)
        rgb_cam.initialControl = ctrl

        # Control input (so we can push a manual fallback later)
        rgb_ctrl_in = pipeline.create(dai.node.XLinkIn)
        rgb_ctrl_in.setStreamName("rgb_ctrl")
        rgb_ctrl_in.out.link(rgb_cam.inputControl)

        # --- Mono cams for stereo ---
        left_cam = pipeline.create(dai.node.MonoCamera)
        left_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        left_cam.setResolution(getattr(dai.MonoCameraProperties.SensorResolution, CAMERA_RESOLUTION['mono']))

        right_cam = pipeline.create(dai.node.MonoCamera)
        right_cam.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        right_cam.setResolution(getattr(dai.MonoCameraProperties.SensorResolution, CAMERA_RESOLUTION['mono']))

        # --- Stereo depth ---
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(getattr(dai.node.StereoDepth.PresetMode, STEREO_DEPTH_CONFIG['preset']))
        stereo.setLeftRightCheck(STEREO_DEPTH_CONFIG['left_right_check'])
        stereo.setExtendedDisparity(STEREO_DEPTH_CONFIG['extended_disparity'])
        stereo.setSubpixel(STEREO_DEPTH_CONFIG['subpixel'])
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align to RGB

        # --- Outputs ---
        rgb_out = pipeline.create(dai.node.XLinkOut); rgb_out.setStreamName("rgb")
        depth_out = pipeline.create(dai.node.XLinkOut); depth_out.setStreamName("depth")

        # Link ISP, not video (ISP gives properly processed frames)
        rgb_cam.isp.link(rgb_out.input)
        left_cam.out.link(stereo.left)
        right_cam.out.link(stereo.right)
        stereo.depth.link(depth_out.input)

        print("✓ Camera pipeline created successfully (ISP output, AE/AWB enabled)")
        return pipeline
    except Exception as e:
        print(f"✗ Failed to create pipeline: {e}")
        return None

def capture_images(device, pipeline):
    try:
        print("Starting camera pipeline...")
        device.startPipeline(pipeline)

        rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        ctrl_queue = device.getInputQueue("rgb_ctrl")

        print(f"Waiting {STABILIZE_SECS}s for AE/AWB to stabilize.")
        time.sleep(STABILIZE_SECS)

        rgb_frame, depth_frame = None, None
        start_time = time.time(); timeout = 10
        while time.time() - start_time < timeout:
            rgb_packet = rgb_queue.tryGet()
            depth_packet = depth_queue.tryGet()

            if rgb_packet is not None:
                rgb_frame = rgb_packet.getCvFrame()
                print("✓ RGB frame captured")
            if depth_packet is not None:
                depth_frame = depth_packet.getFrame()  # 16-bit depth
                print("✓ Depth frame captured")

            if rgb_frame is not None and depth_frame is not None:
                break
            time.sleep(0.05)

        if rgb_frame is None or depth_frame is None:
            print("✗ Failed to capture frames within timeout")
            return None, None

        # Brightness check — if still too dark, push a manual exposure once
        mean_brightness = float(rgb_frame.mean())
        print(f"INFO: mean RGB brightness = {mean_brightness:.1f}")
        if mean_brightness < FALLBACK_BRIGHTNESS_MEAN:
            print("⚠️ Frame looks dark; pushing manual exposure fallback...")
            manual = dai.CameraControl()
            manual.setManualExposure(MANUAL_EXPOSURE_US, MANUAL_ISO)
            manual.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            ctrl_queue.send(manual)
            # give 3A a moment to apply and re-grab one more frame
            time.sleep(0.3)
            pkt = rgb_queue.get()  # blocking get a fresh frame
            rgb_frame = pkt.getCvFrame()
            print(f"✓ New mean brightness = {rgb_frame.mean():.1f}")

        return rgb_frame, depth_frame
    except Exception as e:
        print(f"✗ Error capturing images: {e}")
        return None, None

def save_images(rgb_frame, depth_frame):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save RGB (BGR 8-bit)
        ok_rgb = cv2.imwrite(RGB_IMAGE_PATH, rgb_frame)
        if not ok_rgb:
            raise RuntimeError("Failed to write RGB PNG")
        print(f"✓ RGB image saved: {RGB_IMAGE_PATH}")

        # Save RAW 16-bit depth (no normalization)
        ok = cv2.imwrite(DEPTH_MAP_PATH, depth_frame)
        if not ok:
            raise RuntimeError("Failed to write 16-bit depth PNG")
        print(f"✓ 16-bit depth saved: {DEPTH_MAP_PATH}")

        # Optional preview for debugging (8-bit)
        depth_preview = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        preview_path = f"depth_preview_{timestamp}.png"
        cv2.imwrite(preview_path, depth_preview)
        print(f"✓ Depth preview saved: {preview_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving images: {e}")
        return False

def main():
    print("=" * 50)
    print("OAK-D Lite Image Capture")
    print("=" * 50)

    print("\n1. Detecting camera...")
    device_info = detect_camera(dai)
    if not device_info: sys.exit(1)

    print("\n2. Creating camera pipeline...")
    pipeline = create_camera_pipeline(dai)
    if not pipeline: sys.exit(1)

    print("\n3. Connecting to camera...")
    try:
        with dai.Device(device_info) as device:
            print("✓ Connected to camera successfully")
            print("\n4. Capturing images...")
            rgb_frame, depth_frame = capture_images(device, pipeline)
            if rgb_frame is not None and depth_frame is not None:
                print(f"✓ Captured RGB:  {rgb_frame.shape}")
                print(f"✓ Captured DEPTH:{depth_frame.shape}")
                print("\n5. Saving images...")
                if save_images(rgb_frame, depth_frame):
                    print("\n🎉 Image capture completed successfully!")
                    print("Generated files:")
                    print(f"  - {RGB_IMAGE_PATH}")
                    print(f"  - {DEPTH_MAP_PATH} (16-bit)")
                else:
                    print("\n⚠️ Image capture completed with errors saving images.")
            else:
                print("✗ Failed to capture images from camera.")
    except Exception as e:
        print(f"✗ Error connecting to camera: {e}")
        sys.exit(1)

if __name__ == "__main__":
    t0 = time.time(); main(); print(f"Script execution time: {time.time()-t0:.2f} s")
