use anyhow::Result;
use image::{ImageBuffer, Rgb, RgbImage};
use std::path::Path;
use tracing::{debug, info, warn};

pub struct WebcamCapture {
    width: u32,
    height: u32,
    frame_count: u32,
}

impl WebcamCapture {
    pub fn new(device_path: Option<&str>) -> Result<Self> {
        let device = device_path.unwrap_or("/dev/video0");
        
        if !Path::new(device).exists() {
            warn!("Webcam device {} not found, creating simulated capture", device);
        } else {
            info!("Using webcam device: {}", device);
        }
        
        Ok(Self {
            width: 640,
            height: 480,
            frame_count: 0,
        })
    }

    pub fn capture_frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        self.frame_count += 1;
        
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
        
        debug!("Generated simulated frame {}: {}x{}", self.frame_count, self.width, self.height);
        
        Ok(buffer)
    }

    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn stop(&mut self) -> Result<()> {
        info!("Simulated camera stream stopped");
        Ok(())
    }
}