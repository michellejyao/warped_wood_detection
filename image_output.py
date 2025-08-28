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

def detect_camera(dai):
    """Detect if OAK-D Lite camera is connected."""
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
    """Create a pipeline for RGB and depth map capture."""
    try:
        pipeline = dai.Pipeline()

        # RGB camera
        rgb_cam = pipeline.create(dai.node.ColorCamera)
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        rgb_cam.setResolution(getattr(dai.ColorCameraProperties.SensorResolution, CAMERA_RESOLUTION['rgb']))
        rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        rgb_cam.setInterleaved(False)

        # Stereo mono cams
        left_cam = pipeline.create(dai.node.MonoCamera)
        left_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        left_cam.setResolution(getattr(dai.MonoCameraProperties.SensorResolution, CAMERA_RESOLUTION['mono']))

        right_cam = pipeline.create(dai.node.MonoCamera)
        right_cam.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        right_cam.setResolution(getattr(dai.MonoCameraProperties.SensorResolution, CAMERA_RESOLUTION['mono']))

        # Stereo depth
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(getattr(dai.node.StereoDepth.PresetMode, STEREO_DEPTH_CONFIG['preset']))
        stereo.setLeftRightCheck(STEREO_DEPTH_CONFIG['left_right_check'])
        stereo.setExtendedDisparity(STEREO_DEPTH_CONFIG['extended_disparity'])
        stereo.setSubpixel(STEREO_DEPTH_CONFIG['subpixel'])

        # Align depth to RGB viewpoint
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        # Outputs
        rgb_out = pipeline.create(dai.node.XLinkOut); rgb_out.setStreamName("rgb")
        depth_out = pipeline.create(dai.node.XLinkOut); depth_out.setStreamName("depth")

        # Link
        rgb_cam.video.link(rgb_out.input)
        left_cam.out.link(stereo.left)
        right_cam.out.link(stereo.right)
        stereo.depth.link(depth_out.input)

        print("✓ Camera pipeline created successfully")
        return pipeline
    except Exception as e:
        print(f"✗ Failed to create pipeline: {e}")
        return None

def capture_images(device, pipeline):
    """Capture RGB and depth images from the camera."""
    try:
        print("Starting camera pipeline...")
        device.startPipeline(pipeline)

        rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        print("Waiting for camera to stabilize.")
        time.sleep(2)

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
            time.sleep(0.1)

        if rgb_frame is None or depth_frame is None:
            print("✗ Failed to capture frames within timeout")
            return None, None
        return rgb_frame, depth_frame
    except Exception as e:
        print(f"✗ Error capturing images: {e}")
        return None, None

def save_images(rgb_frame, depth_frame):
    """Save captured images to disk."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save RGB
        cv2.imwrite(RGB_IMAGE_PATH, rgb_frame)
        print(f"✓ RGB image saved: {RGB_IMAGE_PATH}")

        # Save RAW 16-bit depth to DEPTH_MAP_PATH (no normalization!)
        ok = cv2.imwrite(DEPTH_MAP_PATH, depth_frame)
        if not ok:
            raise RuntimeError("Failed to write 16-bit depth PNG")
        print(f"✓ 16-bit depth saved: {DEPTH_MAP_PATH}")

        # Also save a preview for debugging (8-bit)
        depth_preview = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        preview_path = f"depth_preview_{timestamp}.png"
        cv2.imwrite(preview_path, depth_preview)
        print(f"✓ Depth preview saved: {preview_path}")

        # Optional: raw .npy snapshot for debugging
        raw_npy = f"depth_raw_{timestamp}.npy"
        np.save(raw_npy, depth_frame)
        print(f"✓ Raw depth snapshot saved: {raw_npy}")
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
