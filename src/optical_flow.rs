use image::{ImageBuffer, Rgb, Luma, GrayImage};
use std::collections::HashMap;

const LUCAS_KANADE_WINDOW_SIZE: usize = 15;
const MAX_PYRAMIDS: usize = 3;
const OPTICAL_FLOW_THRESHOLD: f32 = 1.0;

#[derive(Clone, Debug)]
pub struct FlowVector {
    pub u: f32, // horizontal flow
    pub v: f32, // vertical flow
    pub confidence: f32,
}

impl FlowVector {
    pub fn new(u: f32, v: f32, confidence: f32) -> Self {
        Self { u, v, confidence }
    }

    pub fn magnitude(&self) -> f32 {
        (self.u * self.u + self.v * self.v).sqrt()
    }

    pub fn is_significant(&self) -> bool {
        self.magnitude() > OPTICAL_FLOW_THRESHOLD && self.confidence > 0.5
    }
}

pub struct OpticalFlowTracker {
    prev_frame: Option<GrayImage>,
    flow_field: HashMap<(u32, u32), FlowVector>,
    width: u32,
    height: u32,
}

impl OpticalFlowTracker {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            prev_frame: None,
            flow_field: HashMap::new(),
            width,
            height,
        }
    }

    pub fn compute_flow(&mut self, current_frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> HashMap<(u32, u32), FlowVector> {
        // Convert RGB to grayscale
        let gray_current = self.rgb_to_gray(current_frame);

        let flow = if let Some(ref prev_gray) = self.prev_frame {
            self.lucas_kanade_flow(prev_gray, &gray_current)
        } else {
            HashMap::new()
        };

        self.prev_frame = Some(gray_current);
        self.flow_field = flow.clone();
        flow
    }

    fn rgb_to_gray(&self, rgb_img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> GrayImage {
        ImageBuffer::from_fn(rgb_img.width(), rgb_img.height(), |x, y| {
            let rgb = rgb_img.get_pixel(x, y);
            // Standard RGB to grayscale conversion
            let gray_val = (0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32) as u8;
            Luma([gray_val])
        })
    }

    fn lucas_kanade_flow(&self, prev: &GrayImage, curr: &GrayImage) -> HashMap<(u32, u32), FlowVector> {
        let mut flow_vectors = HashMap::new();
        let window_half = LUCAS_KANADE_WINDOW_SIZE / 2;

        // Sample points across the image (not every pixel for performance)
        let step = 8; // Process every 8th pixel
        
        for y in (window_half as u32..self.height - window_half as u32).step_by(step) {
            for x in (window_half as u32..self.width - window_half as u32).step_by(step) {
                if let Some(flow) = self.compute_lucas_kanade_at_point(prev, curr, x, y) {
                    flow_vectors.insert((x, y), flow);
                }
            }
        }

        flow_vectors
    }

    fn compute_lucas_kanade_at_point(&self, prev: &GrayImage, curr: &GrayImage, x: u32, y: u32) -> Option<FlowVector> {
        let window_half = LUCAS_KANADE_WINDOW_SIZE / 2;
        let mut sum_ix2 = 0.0f32;
        let mut sum_iy2 = 0.0f32;
        let mut sum_ixy = 0.0f32;
        let mut sum_ixt = 0.0f32;
        let mut sum_iyt = 0.0f32;

        // Calculate gradients and temporal differences within the window
        for wy in 0..LUCAS_KANADE_WINDOW_SIZE {
            for wx in 0..LUCAS_KANADE_WINDOW_SIZE {
                let px = x as i32 + wx as i32 - window_half as i32;
                let py = y as i32 + wy as i32 - window_half as i32;

                if px < 1 || py < 1 || px >= (self.width - 1) as i32 || py >= (self.height - 1) as i32 {
                    continue;
                }

                let px = px as u32;
                let py = py as u32;

                // Spatial gradients (using central differences)
                let ix = (self.get_pixel_safe(prev, px + 1, py) as f32 - self.get_pixel_safe(prev, px - 1, py) as f32) / 2.0;
                let iy = (self.get_pixel_safe(prev, px, py + 1) as f32 - self.get_pixel_safe(prev, px, py - 1) as f32) / 2.0;

                // Temporal gradient
                let it = self.get_pixel_safe(curr, px, py) as f32 - self.get_pixel_safe(prev, px, py) as f32;

                // Accumulate for least squares solution
                sum_ix2 += ix * ix;
                sum_iy2 += iy * iy;
                sum_ixy += ix * iy;
                sum_ixt += ix * it;
                sum_iyt += iy * it;
            }
        }

        // Solve the Lucas-Kanade equation: A * [u, v]^T = b
        // Where A = [[sum_ix2, sum_ixy], [sum_ixy, sum_iy2]]
        // and b = [-sum_ixt, -sum_iyt]

        let det = sum_ix2 * sum_iy2 - sum_ixy * sum_ixy;
        
        if det.abs() < 1e-6 {
            return None; // Singular matrix, no reliable flow
        }

        let u = (sum_iy2 * (-sum_ixt) - sum_ixy * (-sum_iyt)) / det;
        let v = (sum_ix2 * (-sum_iyt) - sum_ixy * (-sum_ixt)) / det;

        // Calculate confidence based on eigenvalues of the structure tensor
        let trace = sum_ix2 + sum_iy2;
        let det_st = sum_ix2 * sum_iy2 - sum_ixy * sum_ixy;
        
        let confidence = if trace > 0.0 {
            (det_st / (trace * trace)).min(1.0).max(0.0)
        } else {
            0.0
        };

        Some(FlowVector::new(u, v, confidence))
    }

    fn get_pixel_safe(&self, img: &GrayImage, x: u32, y: u32) -> u8 {
        if x < img.width() && y < img.height() {
            img.get_pixel(x, y)[0]
        } else {
            0
        }
    }

    pub fn propagate_mask(&self, prev_mask: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut propagated_mask = ImageBuffer::new(self.width, self.height);

        // Initialize with previous mask
        for y in 0..self.height {
            for x in 0..self.width {
                let prev_pixel = prev_mask.get_pixel(x, y)[0];
                propagated_mask.put_pixel(x, y, Luma([prev_pixel]));
            }
        }

        // Apply flow vectors to propagate foreground regions
        for ((fx, fy), flow) in &self.flow_field {
            if !flow.is_significant() {
                continue;
            }

            let prev_value = prev_mask.get_pixel(*fx, *fy)[0];
            if prev_value > 128 { // If was foreground
                // Calculate new position
                let new_x = (*fx as f32 + flow.u).round() as i32;
                let new_y = (*fy as f32 + flow.v).round() as i32;

                if new_x >= 0 && new_x < self.width as i32 && new_y >= 0 && new_y < self.height as i32 {
                    let new_x = new_x as u32;
                    let new_y = new_y as u32;

                    // Blend with confidence
                    let current_value = propagated_mask.get_pixel(new_x, new_y)[0];
                    let blended_value = (flow.confidence * prev_value as f32 + (1.0 - flow.confidence) * current_value as f32) as u8;
                    propagated_mask.put_pixel(new_x, new_y, Luma([blended_value]));
                }
            }
        }

        propagated_mask
    }

    pub fn get_motion_magnitude_map(&self) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut motion_map = ImageBuffer::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let magnitude = if let Some(flow) = self.flow_field.get(&(x, y)) {
                    (flow.magnitude() * 10.0).min(255.0) as u8
                } else {
                    0
                };
                motion_map.put_pixel(x, y, Luma([magnitude]));
            }
        }

        motion_map
    }

    pub fn reset(&mut self) {
        self.prev_frame = None;
        self.flow_field.clear();
    }

    pub fn interpolate_flow(&self, x: f32, y: f32) -> Option<FlowVector> {
        // Bilinear interpolation of flow vectors
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        // Get flow vectors at the four corners
        let f00 = self.flow_field.get(&(x0, y0));
        let f10 = self.flow_field.get(&(x1, y0));
        let f01 = self.flow_field.get(&(x0, y1));
        let f11 = self.flow_field.get(&(x1, y1));

        // If any corner is missing, return None
        if f00.is_none() || f10.is_none() || f01.is_none() || f11.is_none() {
            return None;
        }

        let f00 = f00.unwrap();
        let f10 = f10.unwrap();
        let f01 = f01.unwrap();
        let f11 = f11.unwrap();

        // Bilinear interpolation
        let u = f00.u * (1.0 - dx) * (1.0 - dy) +
                f10.u * dx * (1.0 - dy) +
                f01.u * (1.0 - dx) * dy +
                f11.u * dx * dy;

        let v = f00.v * (1.0 - dx) * (1.0 - dy) +
                f10.v * dx * (1.0 - dy) +
                f01.v * (1.0 - dx) * dy +
                f11.v * dx * dy;

        let confidence = f00.confidence * (1.0 - dx) * (1.0 - dy) +
                        f10.confidence * dx * (1.0 - dy) +
                        f01.confidence * (1.0 - dx) * dy +
                        f11.confidence * dx * dy;

        Some(FlowVector::new(u, v, confidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_vector_magnitude() {
        let flow = FlowVector::new(3.0, 4.0, 0.8);
        assert_eq!(flow.magnitude(), 5.0);
    }

    #[test]
    fn test_optical_flow_tracker_creation() {
        let tracker = OpticalFlowTracker::new(640, 480);
        assert_eq!(tracker.width, 640);
        assert_eq!(tracker.height, 480);
        assert!(tracker.prev_frame.is_none());
    }

    #[test]
    fn test_rgb_to_gray_conversion() {
        let tracker = OpticalFlowTracker::new(2, 2);
        let rgb_img = ImageBuffer::from_fn(2, 2, |x, y| {
            if x == 0 && y == 0 {
                Rgb([255, 0, 0]) // Red
            } else {
                Rgb([0, 255, 0]) // Green
            }
        });

        let gray_img = tracker.rgb_to_gray(&rgb_img);
        
        // Red pixel should be darker than green pixel due to RGB->Gray conversion weights
        let red_gray = gray_img.get_pixel(0, 0)[0];
        let green_gray = gray_img.get_pixel(1, 0)[0];
        
        assert!(red_gray < green_gray);
    }
}