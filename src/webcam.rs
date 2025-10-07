use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb, RgbImage};
use tracing::{debug, info, warn};

use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};

pub struct WebcamCapture {
    camera: Option<Camera>,
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    frame_count: u32,
    simulated: bool,
    flip_horizontal: bool,
}

impl WebcamCapture {
    pub fn new(device_path: Option<&str>, target_width: u32, target_height: u32, flip_horizontal: bool) -> Result<Self> {
        let device = device_path.unwrap_or("/dev/video0");
        
        // Try to create real camera first
        match Self::try_create_camera(device) {
            Ok((camera, width, height)) => {
                info!("Successfully opened webcam device: {} ({}x{}) -> scaling to {}x{}", device, width, height, target_width, target_height);
                Ok(Self {
                    camera: Some(camera),
                    width,
                    height,
                    target_width,
                    target_height,
                    frame_count: 0,
                    simulated: false,
                    flip_horizontal,
                })
            }
            Err(e) => {
                warn!("Failed to open webcam device {}: {} - falling back to simulated camera feed", device, e);
                debug!("Camera initialization failed due to: {:?}", e);
                Ok(Self {
                    camera: None,
                    width: target_width,
                    height: target_height,
                    target_width,
                    target_height,
                    frame_count: 0,
                    simulated: true,
                    flip_horizontal,
                })
            }
        }
    }

    fn try_create_camera(device_path: &str) -> Result<(Camera, u32, u32)> {
        let index = if device_path.starts_with("/dev/video") {
            let num_str = device_path.strip_prefix("/dev/video").unwrap_or("0");
            CameraIndex::Index(num_str.parse().unwrap_or(0))
        } else {
            CameraIndex::Index(0)
        };

        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        
        let mut camera = Camera::new(index, requested)
            .context("Failed to create camera")?;
        
        camera.open_stream()
            .context("Failed to open camera stream")?;

        let resolution = camera.resolution();
        let width = resolution.width();
        let height = resolution.height();

        Ok((camera, width, height))
    }

    pub fn capture_frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        self.frame_count += 1;
        
        let raw_frame = if let Some(camera) = self.camera.as_mut() {
            // Real camera capture
            match Self::capture_real_frame_static(camera, self.width, self.height) {
                Ok(frame) => {
                    debug!("Captured real frame {}: {}x{}", self.frame_count, self.width, self.height);
                    frame
                }
                Err(e) => {
                    warn!("Failed to capture real frame: {}, falling back to simulation", e);
                    debug!("Camera capture error: {:?}", e);
                    self.simulated = true;
                    self.generate_simulated_frame()
                }
            }
        } else {
            // Simulated camera capture (fallback)
            debug!("Generated simulated frame {}: {}x{}", self.frame_count, self.width, self.height);
            self.generate_simulated_frame()
        };

        // Apply horizontal flip if requested
        let processed_frame = if self.flip_horizontal {
            debug!("Applying horizontal flip to frame {}", self.frame_count);
            image::imageops::flip_horizontal(&raw_frame)
        } else {
            raw_frame
        };

        // Scale frame to target dimensions if needed
        if (self.width, self.height) != (self.target_width, self.target_height) {
            let scaled_frame = image::imageops::resize(
                &processed_frame,
                self.target_width,
                self.target_height,
                image::imageops::FilterType::Nearest, // Fastest resize
            );
            debug!("Scaled frame from {}x{} to {}x{}", self.width, self.height, self.target_width, self.target_height);
            Ok(scaled_frame)
        } else {
            Ok(processed_frame)
        }
    }

    fn capture_real_frame_static(camera: &mut Camera, width: u32, height: u32) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let frame = camera.frame()
            .context("Failed to capture camera frame")?;

        let rgb_data = frame.decode_image::<RgbFormat>()
            .context("Failed to decode camera frame")?;

        let buffer = ImageBuffer::from_raw(width, height, rgb_data.into_raw())
            .context("Failed to create image buffer from camera data")?;

        Ok(buffer)
    }

    fn generate_simulated_frame(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        // Use native resolution if camera exists, otherwise use target resolution
        let (frame_width, frame_height) = if self.camera.is_some() {
            (self.width, self.height)
        } else {
            (self.target_width, self.target_height)
        };
        
        let mut buffer = RgbImage::new(frame_width, frame_height);
        
        let time_factor = (self.frame_count as f32 * 0.1).sin();
        let base_color = ((time_factor * 127.0 + 128.0) as u8).clamp(50, 200);
        
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            let distance_from_center = (
                ((x as f32 - frame_width as f32 / 2.0).powi(2) + 
                 (y as f32 - frame_height as f32 / 2.0).powi(2))
            ).sqrt() / (frame_width as f32 / 2.0);
            
            let fade = (1.0 - distance_from_center.clamp(0.0, 1.0)) * 255.0;
            
            *pixel = Rgb([
                (base_color as f32 * fade / 255.0) as u8,
                ((base_color + 50) as f32 * fade / 255.0) as u8,
                ((base_color + 100) as f32 * fade / 255.0) as u8,
            ]);
        }
        
        buffer
    }

    pub fn resolution(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }

    pub fn stop(&mut self) -> Result<()> {
        if let Some(camera) = &mut self.camera {
            camera.stop_stream()
                .context("Failed to stop camera stream")?;
            info!("Real camera stream stopped");
        } else {
            info!("Simulated camera stream stopped");
        }
        Ok(())
    }

}

impl Drop for WebcamCapture {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            warn!("Failed to stop camera during drop: {}", e);
        }
    }
}
