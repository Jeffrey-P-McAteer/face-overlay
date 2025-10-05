use anyhow::Result;
use image::{ImageBuffer, Rgb, Rgba};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::segmentation::SegmentationModel;
use crate::webcam::WebcamCapture;

/// Frame data flowing through the pipeline
#[derive(Clone)]
pub struct FrameData {
    pub image: ImageBuffer<Rgb<u8>, Vec<u8>>,
    pub timestamp: Instant,
    pub frame_id: u64,
}

/// Processed frame with segmentation applied
#[derive(Clone)]
pub struct ProcessedFrame {
    pub image: ImageBuffer<Rgba<u8>, Vec<u8>>,
    pub timestamp: Instant,
    pub frame_id: u64,
    pub processing_time_ms: u64,
}

/// Pipeline configuration
#[derive(Clone)]
pub struct PipelineConfig {
    pub target_fps: u32,
    pub max_queue_size: usize,
    pub ai_processing_interval: u32, // Process every Nth frame for AI
    pub frame_drop_threshold: usize, // Drop frames if queue exceeds this
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            target_fps: 30,
            max_queue_size: 5,
            ai_processing_interval: 1, // Process every frame initially
            frame_drop_threshold: 3,
        }
    }
}

/// Multi-threaded processing pipeline
pub struct ProcessingPipeline {
    config: PipelineConfig,
    
    // Channels for inter-task communication
    raw_frame_tx: mpsc::Sender<FrameData>,
    
    // Shared state for latest frame (non-blocking access)
    latest_frame: Arc<RwLock<Option<ProcessedFrame>>>,
    
    // Task handles
    ai_task: tokio::task::JoinHandle<Result<()>>,
    frame_updater_task: tokio::task::JoinHandle<()>,
    
    // Camera stays on main thread to avoid Send issues
    webcam: WebcamCapture,
    frame_counter: u64,
    last_capture_time: std::time::Instant,
}

impl ProcessingPipeline {
    pub async fn new(
        webcam: WebcamCapture,
        segmentation_model: Option<SegmentationModel>,
        config: PipelineConfig,
    ) -> Result<Self> {
        let (raw_frame_tx, raw_frame_rx) = mpsc::channel::<FrameData>(config.max_queue_size);
        let (processed_frame_tx, processed_frame_rx) = mpsc::channel::<ProcessedFrame>(config.max_queue_size);
        
        let latest_frame = Arc::new(RwLock::new(None));
        let latest_frame_clone = latest_frame.clone();
        
        info!("Starting multi-threaded processing pipeline");
        info!("Configuration: target_fps={}, max_queue_size={}, ai_interval={}", 
              config.target_fps, config.max_queue_size, config.ai_processing_interval);
        
        // Spawn AI processing task
        let ai_config = config.clone();
        let ai_task = tokio::spawn(async move {
            Self::ai_processing_task(raw_frame_rx, processed_frame_tx, segmentation_model, ai_config).await
        });
        
        // Spawn frame update task (updates latest_frame for non-blocking access)
        let frame_updater_task = tokio::spawn(async move {
            if let Err(e) = Self::frame_update_task(processed_frame_rx, latest_frame_clone).await {
                tracing::error!("Frame update task error: {}", e);
            }
        });
        
        Ok(Self {
            config,
            raw_frame_tx,
            latest_frame,
            ai_task,
            frame_updater_task,
            webcam,
            frame_counter: 0,
            last_capture_time: std::time::Instant::now(),
        })
    }
    
    
    /// AI processing task - processes frames with segmentation
    async fn ai_processing_task(
        mut rx: mpsc::Receiver<FrameData>,
        tx: mpsc::Sender<ProcessedFrame>,
        mut segmentation_model: Option<SegmentationModel>,
        config: PipelineConfig,
    ) -> Result<()> {
        info!("AI processing task started");
        let mut frame_counter = 0u32;
        
        while let Some(frame_data) = rx.recv().await {
            let processing_start = Instant::now();
            frame_counter += 1;
            
            // Skip AI processing for some frames to maintain performance
            let should_process_ai = frame_counter % config.ai_processing_interval == 0;
            
            let processed_image = if should_process_ai && segmentation_model.is_some() {
                debug!("Processing frame {} with AI segmentation", frame_data.frame_id);
                
                match segmentation_model.as_mut().unwrap().segment_foreground(&frame_data.image) {
                    Ok(segmented) => segmented,
                    Err(e) => {
                        warn!("AI segmentation failed for frame {}: {} - using raw frame", frame_data.frame_id, e);
                        Self::convert_rgb_to_rgba(&frame_data.image)
                    }
                }
            } else {
                // Fast path: just convert RGB to RGBA without AI processing
                debug!("Fast-path processing frame {} (no AI)", frame_data.frame_id);
                Self::convert_rgb_to_rgba(&frame_data.image)
            };
            
            let processing_time = processing_start.elapsed();
            let processed_frame = ProcessedFrame {
                image: processed_image,
                timestamp: frame_data.timestamp,
                frame_id: frame_data.frame_id,
                processing_time_ms: processing_time.as_millis() as u64,
            };
            
            // Send processed frame - use blocking send to ensure delivery
            match tx.send(processed_frame).await {
                Ok(_) => {
                    info!("AI processed frame {} in {}ms - sent to renderer", frame_data.frame_id, processing_time.as_millis());
                }
                Err(_) => {
                    info!("AI task stopping - channel closed");
                    break;
                }
            }
        }
        
        info!("AI processing task finished");
        Ok(())
    }
    
    /// Frame update task - maintains latest frame for non-blocking access
    async fn frame_update_task(
        mut rx: mpsc::Receiver<ProcessedFrame>,
        latest_frame: Arc<RwLock<Option<ProcessedFrame>>>,
    ) -> Result<()> {
        info!("Frame update task started");
        
        while let Some(processed_frame) = rx.recv().await {
            // Update latest frame (non-blocking write)
            {
                let mut latest = latest_frame.write().await;
                *latest = Some(processed_frame.clone());
            }
            
            info!("Frame update: latest frame now {} ({}ms processing) - ready for rendering", 
                   processed_frame.frame_id, processed_frame.processing_time_ms);
        }
        
        info!("Frame update task finished");
        Ok(())
    }
    
    /// Capture a frame from the camera and send it to AI processing
    /// This should be called from the main thread regularly
    pub async fn capture_frame(&mut self) -> Result<()> {
        let frame_duration = Duration::from_millis(1000 / self.config.target_fps as u64);
        let now = std::time::Instant::now();
        
        // Debug: Show capture_frame is being called
        if self.frame_counter % 100 == 0 {
            debug!("capture_frame called - frame_counter at {}", self.frame_counter);
        }
        
        // Maintain target FPS
        let elapsed = now.duration_since(self.last_capture_time);
        if elapsed < frame_duration {
            debug!("Frame rate limiting: elapsed {}ms < target {}ms", 
                   elapsed.as_millis(), frame_duration.as_millis());
            return Ok(());
        }
        
        debug!("Attempting to capture frame {} (target_fps={})", self.frame_counter + 1, self.config.target_fps);
        
        match self.webcam.capture_frame() {
            Ok(frame) => {
                self.frame_counter += 1;
                let (width, height) = frame.dimensions();
                let frame_data = FrameData {
                    image: frame,
                    timestamp: tokio::time::Instant::now(),
                    frame_id: self.frame_counter,
                };
                
                // Non-blocking send - drop frame if queue is full
                match self.raw_frame_tx.try_send(frame_data) {
                    Ok(_) => {
                        info!("Camera captured frame {} ({}x{}) - sent to AI", 
                              self.frame_counter, width, height);
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        warn!("Frame {} dropped - AI processing queue full", self.frame_counter);
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        warn!("Camera stopping - AI processing channel closed");
                    }
                }
                
                self.last_capture_time = now;
            }
            Err(e) => {
                warn!("Camera capture error: {} - retrying", e);
            }
        }
        
        Ok(())
    }
    
    /// Get the latest processed frame without blocking
    pub async fn get_latest_frame(&self) -> Option<ProcessedFrame> {
        let latest = self.latest_frame.read().await;
        latest.clone()
    }
    
    /// Convert RGB to RGBA (fast path without AI)
    fn convert_rgb_to_rgba(rgb_image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let (width, height) = rgb_image.dimensions();
        ImageBuffer::from_fn(width, height, |x, y| {
            let pixel = rgb_image.get_pixel(x, y);
            Rgba([pixel[0], pixel[1], pixel[2], 255])
        })
    }
    
    /// Get pipeline statistics
    pub fn get_stats(&self) -> PipelineStats {
        PipelineStats {
            frame_count: self.frame_counter,
            camera_queue_size: self.raw_frame_tx.capacity() - self.raw_frame_tx.capacity(),
        }
    }
    
    /// Shutdown the pipeline gracefully
    pub async fn shutdown(self) -> Result<()> {
        info!("Shutting down processing pipeline");
        
        // Drop the sender to signal tasks to stop
        drop(self.raw_frame_tx);
        
        // Wait for tasks to complete
        if let Err(e) = self.ai_task.await? {
            warn!("AI task error during shutdown: {}", e);
        }
        
        if let Err(e) = self.frame_updater_task.await {
            warn!("Frame updater task join error during shutdown: {}", e);
        }
        
        info!("Processing pipeline shutdown complete");
        Ok(())
    }
}

#[derive(Debug)]
pub struct PipelineStats {
    pub frame_count: u64,
    pub camera_queue_size: usize,
}