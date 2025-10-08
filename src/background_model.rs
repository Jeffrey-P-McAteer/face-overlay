use image::{ImageBuffer, Rgb, Luma};
use std::collections::VecDeque;

const MAX_GAUSSIANS: usize = 5;
const DEFAULT_LEARNING_RATE: f32 = 0.01;
const BACKGROUND_THRESHOLD: f32 = 0.7;
const VARIANCE_THRESHOLD: f32 = 2.5;
const INITIAL_VARIANCE: f32 = 15.0;
const NOISE_SIGMA: f32 = 30.0;

#[derive(Clone, Debug)]
pub struct Gaussian {
    pub weight: f32,
    pub mean: [f32; 3], // RGB mean
    pub variance: [f32; 3], // RGB variance
    pub last_update: u32,
}

impl Gaussian {
    fn new() -> Self {
        Self {
            weight: 0.0,
            mean: [0.0, 0.0, 0.0],
            variance: [INITIAL_VARIANCE, INITIAL_VARIANCE, INITIAL_VARIANCE],
            last_update: 0,
        }
    }

    fn new_from_pixel(pixel: &[u8; 3]) -> Self {
        Self {
            weight: DEFAULT_LEARNING_RATE,
            mean: [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32],
            variance: [INITIAL_VARIANCE, INITIAL_VARIANCE, INITIAL_VARIANCE],
            last_update: 0,
        }
    }

    fn matches(&self, pixel: &[u8; 3]) -> bool {
        for i in 0..3 {
            let diff = (pixel[i] as f32 - self.mean[i]).abs();
            if diff > VARIANCE_THRESHOLD * self.variance[i].sqrt() {
                return false;
            }
        }
        true
    }

    fn update(&mut self, pixel: &[u8; 3], learning_rate: f32, frame_count: u32) {
        self.weight = (1.0 - learning_rate) * self.weight + learning_rate;
        
        for i in 0..3 {
            let pixel_val = pixel[i] as f32;
            let rho = learning_rate / self.weight;
            
            self.mean[i] = (1.0 - rho) * self.mean[i] + rho * pixel_val;
            let diff = pixel_val - self.mean[i];
            self.variance[i] = (1.0 - rho) * self.variance[i] + rho * diff * diff;
            
            // Prevent variance from becoming too small
            self.variance[i] = self.variance[i].max(4.0);
        }
        
        self.last_update = frame_count;
    }

    fn decay(&mut self, learning_rate: f32) {
        self.weight = (1.0 - learning_rate) * self.weight;
    }
}

#[derive(Clone)]
pub struct PixelGMM {
    gaussians: Vec<Gaussian>,
    frame_count: u32,
}

impl PixelGMM {
    fn new() -> Self {
        Self {
            gaussians: Vec::new(),
            frame_count: 0,
        }
    }

    pub fn update(&mut self, pixel: &[u8; 3], learning_rate: f32) -> bool {
        self.frame_count += 1;
        
        // Find matching Gaussian
        let mut matched_idx = None;
        for (i, gaussian) in self.gaussians.iter().enumerate() {
            if gaussian.matches(pixel) {
                matched_idx = Some(i);
                break;
            }
        }

        let mut is_background = false;

        if let Some(idx) = matched_idx {
            // Update matching Gaussian
            self.gaussians[idx].update(pixel, learning_rate, self.frame_count);
            
            // Check if this Gaussian represents background
            is_background = self.is_background_gaussian(idx);
        } else {
            // No match found - create new Gaussian or replace least probable
            if self.gaussians.len() < MAX_GAUSSIANS {
                self.gaussians.push(Gaussian::new_from_pixel(pixel));
            } else {
                // Replace least probable Gaussian
                let min_idx = self.gaussians.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.weight.partial_cmp(&b.weight).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                
                self.gaussians[min_idx] = Gaussian::new_from_pixel(pixel);
            }
        }

        // Decay weights of all non-matching Gaussians
        for (i, gaussian) in self.gaussians.iter_mut().enumerate() {
            if Some(i) != matched_idx {
                gaussian.decay(learning_rate);
            }
        }

        // Sort by weight (most probable first)
        self.gaussians.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());

        // Remove Gaussians with very low weights
        self.gaussians.retain(|g| g.weight > 0.001);

        is_background
    }

    fn is_background_gaussian(&self, idx: usize) -> bool {
        if idx >= self.gaussians.len() {
            return false;
        }

        // Calculate cumulative weight to determine background components
        let mut cumulative_weight = 0.0;
        for (i, gaussian) in self.gaussians.iter().enumerate() {
            cumulative_weight += gaussian.weight;
            if i == idx && cumulative_weight > BACKGROUND_THRESHOLD {
                return true;
            }
            if cumulative_weight > BACKGROUND_THRESHOLD {
                break;
            }
        }

        false
    }

    pub fn classify_pixel(&self, pixel: &[u8; 3]) -> bool {
        let mut cumulative_weight = 0.0;
        
        for gaussian in &self.gaussians {
            cumulative_weight += gaussian.weight;
            
            if gaussian.matches(pixel) && cumulative_weight <= BACKGROUND_THRESHOLD {
                return true; // Background
            }
            
            if cumulative_weight > BACKGROUND_THRESHOLD {
                break;
            }
        }
        
        false // Foreground
    }
}

pub struct BackgroundModel {
    pixel_models: Vec<PixelGMM>,
    width: u32,
    height: u32,
    learning_rate: f32,
    frame_count: u32,
    history: VecDeque<ImageBuffer<Luma<u8>, Vec<u8>>>,
    max_history: usize,
}

impl BackgroundModel {
    pub fn new(width: u32, height: u32) -> Self {
        let pixel_count = (width * height) as usize;
        let mut pixel_models = Vec::with_capacity(pixel_count);
        
        for _ in 0..pixel_count {
            pixel_models.push(PixelGMM::new());
        }

        Self {
            pixel_models,
            width,
            height,
            learning_rate: DEFAULT_LEARNING_RATE,
            frame_count: 0,
            history: VecDeque::new(),
            max_history: 10,
        }
    }

    pub fn update_and_classify(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        self.frame_count += 1;
        
        let mut mask = ImageBuffer::new(self.width, self.height);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel_idx = (y * self.width + x) as usize;
                let rgb_pixel = frame.get_pixel(x, y);
                let pixel_rgb = [rgb_pixel[0], rgb_pixel[1], rgb_pixel[2]];
                
                let is_background = self.pixel_models[pixel_idx].update(&pixel_rgb, self.learning_rate);
                
                // Set mask value: 0 for background, 255 for foreground
                let mask_value = if is_background { 0 } else { 255 };
                mask.put_pixel(x, y, Luma([mask_value]));
            }
        }

        // Apply temporal filtering for consistency
        let filtered_mask = self.apply_temporal_filtering(&mask);
        
        // Store in history
        self.history.push_back(filtered_mask.clone());
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        filtered_mask
    }

    pub fn classify_frame(&self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut mask = ImageBuffer::new(self.width, self.height);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel_idx = (y * self.width + x) as usize;
                let rgb_pixel = frame.get_pixel(x, y);
                let pixel_rgb = [rgb_pixel[0], rgb_pixel[1], rgb_pixel[2]];
                
                let is_background = self.pixel_models[pixel_idx].classify_pixel(&pixel_rgb);
                
                // Set mask value: 0 for background, 255 for foreground
                let mask_value = if is_background { 0 } else { 255 };
                mask.put_pixel(x, y, Luma([mask_value]));
            }
        }

        self.apply_temporal_filtering(&mask)
    }

    fn apply_temporal_filtering(&self, mask: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        if self.history.is_empty() {
            return mask.clone();
        }

        let mut filtered = ImageBuffer::new(self.width, self.height);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let current_pixel = mask.get_pixel(x, y)[0];
                
                // Collect values from history
                let mut values = vec![current_pixel];
                for hist_mask in &self.history {
                    if let Some(hist_pixel) = hist_mask.get_pixel_checked(x, y) {
                        values.push(hist_pixel[0]);
                    }
                }
                
                // Apply median filter for temporal consistency
                values.sort_unstable();
                let median_value = values[values.len() / 2];
                
                filtered.put_pixel(x, y, Luma([median_value]));
            }
        }

        filtered
    }

    pub fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate.clamp(0.001, 0.1);
    }

    pub fn get_frame_count(&self) -> u32 {
        self.frame_count
    }

    pub fn reset(&mut self) {
        for model in &mut self.pixel_models {
            model.gaussians.clear();
            model.frame_count = 0;
        }
        self.frame_count = 0;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_matching() {
        let gaussian = Gaussian {
            weight: 0.5,
            mean: [100.0, 150.0, 200.0],
            variance: [10.0, 10.0, 10.0],
            last_update: 0,
        };

        // Should match similar pixels
        assert!(gaussian.matches(&[98, 152, 198]));
        
        // Should not match very different pixels
        assert!(!gaussian.matches(&[50, 50, 50]));
    }

    #[test]
    fn test_background_model_creation() {
        let model = BackgroundModel::new(640, 480);
        assert_eq!(model.width, 640);
        assert_eq!(model.height, 480);
        assert_eq!(model.pixel_models.len(), 640 * 480);
    }
}