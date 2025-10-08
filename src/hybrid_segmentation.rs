use crate::background_model::BackgroundModel;
use crate::optical_flow::OpticalFlowTracker;
use crate::segmentation::SegmentationModel;
use image::{ImageBuffer, Rgb, Luma};
use anyhow::Result;
use std::time::Instant;

const AI_INFERENCE_INTERVAL: u32 = 30; // Run AI every 30 frames (1 second at 30fps)
const CONFIDENCE_THRESHOLD: f32 = 0.7;
const BLEND_ALPHA: f32 = 0.3; // Weight for traditional methods vs AI

pub struct HybridSegmentationPipeline {
    ai_model: Option<SegmentationModel>,
    background_model: BackgroundModel,
    optical_flow: OpticalFlowTracker,
    frame_count: u32,
    last_ai_mask: Option<ImageBuffer<Luma<u8>, Vec<u8>>>,
    last_ai_frame: u32,
    width: u32,
    height: u32,
    learning_phase_frames: u32,
    is_learning: bool,
}

impl HybridSegmentationPipeline {
    pub fn new(
        ai_model: Option<SegmentationModel>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            ai_model,
            background_model: BackgroundModel::new(width, height),
            optical_flow: OpticalFlowTracker::new(width, height),
            frame_count: 0,
            last_ai_mask: None,
            last_ai_frame: 0,
            width,
            height,
            learning_phase_frames: 120, // Learn background for 4 seconds at 30fps
            is_learning: true,
        }
    }

    pub fn process_frame(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
        self.frame_count += 1;

        // Phase 1: Learning phase - build background model without AI
        if self.is_learning && self.frame_count <= self.learning_phase_frames {
            let learning_rate = 0.05; // Higher learning rate during initialization
            self.background_model.set_learning_rate(learning_rate);
            let mask = self.background_model.update_and_classify(frame);
            
            // Also compute optical flow for initialization
            self.optical_flow.compute_flow(frame);
            
            if self.frame_count == self.learning_phase_frames {
                self.is_learning = false;
                self.background_model.set_learning_rate(0.01); // Lower learning rate after initialization
                tracing::info!("Background learning phase completed after {} frames", self.learning_phase_frames);
            }
            
            return Ok(mask);
        }

        // Phase 2: Hybrid approach
        let should_run_ai = self.should_run_ai_inference();
        
        if should_run_ai {
            self.run_ai_inference(frame)?;
        }

        // Generate masks using different methods
        let traditional_mask = self.generate_traditional_mask(frame)?;
        let flow_mask = self.generate_flow_propagated_mask();
        
        // Combine masks based on confidence and availability
        let final_mask = self.combine_masks(&traditional_mask, &flow_mask)?;
        
        Ok(final_mask)
    }

    fn should_run_ai_inference(&self) -> bool {
        if self.ai_model.is_none() {
            return false;
        }

        // Run AI inference periodically or when motion is detected
        let frames_since_ai = self.frame_count - self.last_ai_frame;
        
        // Force AI inference every N frames
        if frames_since_ai >= AI_INFERENCE_INTERVAL {
            return true;
        }

        // Run AI if significant motion is detected (could indicate scene change)
        let motion_map = self.optical_flow.get_motion_magnitude_map();
        let motion_pixels = motion_map.pixels().filter(|p| p[0] > 50).count();
        let motion_ratio = motion_pixels as f32 / (self.width * self.height) as f32;
        
        motion_ratio > 0.1 // If more than 10% of pixels show significant motion
    }

    fn run_ai_inference(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<()> {
        if let Some(ref mut ai_model) = self.ai_model {
            let start_time = Instant::now();
            
            // Resize frame for AI model
            let resized_frame = image::imageops::resize(
                frame,
                ai_model.input_width as u32,
                ai_model.input_height as u32,
                image::imageops::FilterType::Nearest,
            );

            let ai_mask = ai_model.run_efficient_ai_inference(&resized_frame)?;
            
            // Resize AI mask back to original dimensions
            let full_size_mask = image::imageops::resize(
                &ai_mask,
                self.width,
                self.height,
                image::imageops::FilterType::Nearest,
            );

            self.last_ai_mask = Some(full_size_mask);
            self.last_ai_frame = self.frame_count;

            let inference_time = start_time.elapsed();
            tracing::debug!("AI inference completed in {:?} for frame {}", inference_time, self.frame_count);
        }

        Ok(())
    }

    fn generate_traditional_mask(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
        // Update background model with current frame
        self.background_model.set_learning_rate(0.005); // Slow adaptation
        let mask = self.background_model.update_and_classify(frame);
        
        // Also compute optical flow for next iteration
        self.optical_flow.compute_flow(frame);
        
        Ok(mask)
    }

    fn generate_flow_propagated_mask(&self) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
        self.last_ai_mask.as_ref().map(|last_mask| {
            self.optical_flow.propagate_mask(last_mask)
        })
    }

    fn combine_masks(
        &self,
        traditional_mask: &ImageBuffer<Luma<u8>, Vec<u8>>,
        flow_mask: &Option<ImageBuffer<Luma<u8>, Vec<u8>>>,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let mut combined_mask = ImageBuffer::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let traditional_value = traditional_mask.get_pixel(x, y)[0] as f32 / 255.0;
                
                let final_value = if let Some(ai_propagated) = flow_mask {
                    let ai_value = ai_propagated.get_pixel(x, y)[0] as f32 / 255.0;
                    
                    // Combine traditional and AI-propagated masks
                    // Give more weight to AI when available and recent
                    let frames_since_ai = self.frame_count - self.last_ai_frame;
                    let ai_weight = if frames_since_ai < AI_INFERENCE_INTERVAL / 2 {
                        1.0 - BLEND_ALPHA // Recent AI gets high weight
                    } else {
                        0.5 // Older AI gets medium weight
                    };
                    
                    ai_weight * ai_value + (1.0 - ai_weight) * traditional_value
                } else {
                    // No AI mask available, use traditional only
                    traditional_value
                };

                let pixel_value = (final_value * 255.0).clamp(0.0, 255.0) as u8;
                combined_mask.put_pixel(x, y, Luma([pixel_value]));
            }
        }

        // Apply morphological operations to clean up the mask
        Ok(self.apply_morphological_cleanup(&combined_mask))
    }

    fn apply_morphological_cleanup(&self, mask: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        // Simple erosion followed by dilation to remove noise
        let eroded = self.apply_erosion(mask, 2);
        self.apply_dilation(&eroded, 3)
    }

    fn apply_erosion(&self, mask: &ImageBuffer<Luma<u8>, Vec<u8>>, radius: i32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut result = ImageBuffer::new(self.width, self.height);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let mut min_value = 255u8;
                
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        
                        if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                            let pixel_value = mask.get_pixel(nx as u32, ny as u32)[0];
                            min_value = min_value.min(pixel_value);
                        }
                    }
                }
                
                result.put_pixel(x, y, Luma([min_value]));
            }
        }
        
        result
    }

    fn apply_dilation(&self, mask: &ImageBuffer<Luma<u8>, Vec<u8>>, radius: i32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut result = ImageBuffer::new(self.width, self.height);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let mut max_value = 0u8;
                
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        
                        if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                            let pixel_value = mask.get_pixel(nx as u32, ny as u32)[0];
                            max_value = max_value.max(pixel_value);
                        }
                    }
                }
                
                result.put_pixel(x, y, Luma([max_value]));
            }
        }
        
        result
    }

    pub fn reset(&mut self) {
        self.background_model.reset();
        self.optical_flow.reset();
        self.frame_count = 0;
        self.last_ai_mask = None;
        self.last_ai_frame = 0;
        self.is_learning = true;
    }

    pub fn get_statistics(&self) -> HybridStats {
        HybridStats {
            frame_count: self.frame_count,
            last_ai_frame: self.last_ai_frame,
            frames_since_ai: self.frame_count - self.last_ai_frame,
            is_learning: self.is_learning,
            has_ai_model: self.ai_model.is_some(),
        }
    }

    pub fn force_ai_inference(&mut self) {
        // Force AI inference on next frame by setting last_ai_frame to trigger condition
        self.last_ai_frame = self.frame_count - AI_INFERENCE_INTERVAL;
    }

    pub fn set_learning_rate(&mut self, rate: f32) {
        self.background_model.set_learning_rate(rate);
    }
}

#[derive(Debug, Clone)]
pub struct HybridStats {
    pub frame_count: u32,
    pub last_ai_frame: u32,
    pub frames_since_ai: u32,
    pub is_learning: bool,
    pub has_ai_model: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_pipeline_creation() {
        let pipeline = HybridSegmentationPipeline::new(None, 640, 480);
        assert_eq!(pipeline.width, 640);
        assert_eq!(pipeline.height, 480);
        assert!(pipeline.is_learning);
        assert_eq!(pipeline.frame_count, 0);
    }

    #[test]
    fn test_should_run_ai_inference() {
        let mut pipeline = HybridSegmentationPipeline::new(None, 640, 480);
        
        // Should not run AI if no model
        assert!(!pipeline.should_run_ai_inference());
        
        // Should run AI after interval even with model
        pipeline.ai_model = Some(crate::segmentation::SegmentationModel::new(
            std::path::Path::new("dummy"), 
            crate::segmentation::ModelType::U2Net
        ).unwrap_or_else(|_| panic!("Failed to create model for test")));
        
        pipeline.frame_count = AI_INFERENCE_INTERVAL + 1;
        // Note: This test would need a valid model file to work properly
    }
}