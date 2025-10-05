use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb, RgbImage};
use std::path::Path;
use tracing::{debug, info, warn};

#[cfg(feature = "real-camera")]
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};

pub struct WebcamCapture {
    #[cfg(feature = "real-camera")]
    camera: Option<Camera>,
    width: u32,
    height: u32,
    frame_count: u32,
    simulated: bool,
}

impl WebcamCapture {
    pub fn new(device_path: Option<&str>) -> Result<Self> {
        let device = device_path.unwrap_or("/dev/video0");
        
#[cfg(feature = "real-camera")]
        {
            // Try to create real camera first
            match Self::try_create_camera(device) {
                Ok((camera, width, height)) => {
                    info!("Successfully opened webcam device: {} ({}x{})", device, width, height);
                    Ok(Self {
                        camera: Some(camera),
                        width,
                        height,
                        frame_count: 0,
                        simulated: false,
                    })
                }
                Err(e) => {
                    warn!("Failed to open webcam device {}: {}", device, e);
                    info!("Falling back to simulated camera feed");
                    Ok(Self {
                        camera: None,
                        width: 640,
                        height: 480,
                        frame_count: 0,
                        simulated: true,
                    })
                }
            }
        }
        
        #[cfg(not(feature = "real-camera"))]
        {
            warn!("Real camera support not enabled, using simulated feed");
            info!("To enable real webcam capture, compile with --features real-camera");
            Ok(Self {
                width: 640,
                height: 480,
                frame_count: 0,
                simulated: true,
            })
        }
    }

    #[cfg(feature = "real-camera")]
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
        
#[cfg(feature = "real-camera")]
        if let Some(camera) = &mut self.camera {
            // Real camera capture
            match self.capture_real_frame(camera) {
                Ok(frame) => {
                    debug!("Captured real frame {}: {}x{}", self.frame_count, self.width, self.height);
                    return Ok(frame);
                }
                Err(e) => {
                    warn!("Failed to capture real frame: {}, falling back to simulation", e);
                    self.simulated = true;
                }
            }
        }
        
        // Simulated camera capture (fallback)
        let buffer = self.generate_simulated_frame();
        debug!("Generated simulated frame {}: {}x{}", self.frame_count, self.width, self.height);
        Ok(buffer)
    }

    #[cfg(feature = "real-camera")]
    fn capture_real_frame(&mut self, camera: &mut Camera) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let frame = camera.frame()
            .context("Failed to capture camera frame")?;

        let rgb_data = frame.decode_image::<RgbFormat>()
            .context("Failed to decode camera frame")?;

        let buffer = ImageBuffer::from_raw(self.width, self.height, rgb_data.into_raw())
            .context("Failed to create image buffer from camera data")?;

        Ok(buffer)
    }

    fn generate_simulated_frame(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let mut buffer = RgbImage::new(self.width, self.height);
        
        let time_factor = (self.frame_count as f32 * 0.1).sin();
        let base_color = ((time_factor * 127.0 + 128.0) as u8).clamp(50, 200);
        
        for (x, y, pixel) in buffer.enumerate_pixels_mut() {
            let distance_from_center = (
                ((x as f32 - self.width as f32 / 2.0).powi(2) + 
                 (y as f32 - self.height as f32 / 2.0).powi(2))
            ).sqrt() / (self.width as f32 / 2.0);
            
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
        (self.width, self.height)
    }

    pub fn stop(&mut self) -> Result<()> {
        #[cfg(feature = "real-camera")]
        if let Some(camera) = &mut self.camera {
            camera.stop_stream()
                .context("Failed to stop camera stream")?;
            info!("Real camera stream stopped");
        }
        #[cfg(not(feature = "real-camera"))]
        {
            info!("Simulated camera stream stopped");
        }
        Ok(())
    }

    pub fn is_simulated(&self) -> bool {
        #[cfg(feature = "real-camera")]
        {
            self.simulated || self.camera.is_none()
        }
        #[cfg(not(feature = "real-camera"))]
        {
            true
        }
    }
}

impl Drop for WebcamCapture {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            warn!("Failed to stop camera during drop: {}", e);
        }
    }
}