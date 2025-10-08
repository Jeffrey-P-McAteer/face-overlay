use tracing::warn;
use anyhow::{Context, Result};
use futures_util::StreamExt;
use image::{ImageBuffer, Rgb, Rgba};
use std::path::Path;
// Removed VecDeque import - no longer using inefficient caching

// use ndarray::Array;
use ort::{
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::Value,
};


#[derive(Clone, Debug, PartialEq)]
pub enum ModelType {
    U2Net,
    YoloV8nSeg,
    FastSam,
    MediaPipeSelfie,
    SINet,
}

impl ModelType {
    pub fn filename(&self) -> &'static str {
        match self {
            ModelType::U2Net => "u2net.onnx",
            ModelType::YoloV8nSeg => "yolov8n-seg.onnx", 
            ModelType::FastSam => "fastsam.onnx",
            ModelType::MediaPipeSelfie => "mediapipe_selfie_segmentation.onnx",
            ModelType::SINet => "sinet.onnx",
        }
    }
    
    pub fn input_size(&self) -> (usize, usize) {
        match self {
            ModelType::U2Net => (320, 320),
            ModelType::YoloV8nSeg => (640, 640),
            ModelType::FastSam => (512, 512),
            ModelType::MediaPipeSelfie => (256, 256),  // Optimized for 256x256 input
            ModelType::SINet => (224, 224),  // Optimized for mobile/CPU
        }
    }
    
}

// Removed inefficient MaskCache system - AI runs on every frame for maximum responsiveness

pub struct SegmentationModel {
    session: Session,
    pub input_height: usize,
    pub input_width: usize,
    model_type: ModelType,
    // Pre-allocated buffer for AI data reuse and maximum efficiency
    input_buffer: Vec<f32>,
}

impl SegmentationModel {
    pub fn new(model_path: &Path, model_type: ModelType) -> Result<Self> {
        if !model_path.exists() {
            anyhow::bail!("Model file does not exist at path: {:?}", model_path);
        }
        
        let session = match Self::create_gpu_session(model_path) {
            Ok(gpu_session) => {
                warn!("Successfully loaded GPU ONNX Session!");
                gpu_session
            }
            Err(e) => {
                //std::thread::sleep(std::time::Duration::from_millis(150));
                warn!("Could not load GPU ONNX Session, falling back to CPU: {:?}", e);
                Self::create_cpu_session(model_path)?
            }
        };
        let (input_width, input_height) = model_type.input_size();
        let buffer_size = 3 * input_height * input_width;

        Ok(Self {
            session,
            input_height,
            input_width,
            model_type,
            input_buffer: Vec::with_capacity(buffer_size),
        })
    }

    pub fn segment_foreground(&mut self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let resized_image = image::imageops::resize(image, self.input_width as u32, self.input_height as u32, image::imageops::FilterType::Nearest);
        let mask = self.run_efficient_ai_inference(&resized_image)?;
        self.apply_mask_efficiently(image, &mask)
    }
    
    pub fn run_efficient_ai_inference(&mut self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        self.preprocess_image_efficient(image)?;
        let input_tensor = Value::from_array(([1, 3, self.input_height, self.input_width], self.input_buffer.clone()))?;
        let input_name = self.session.inputs[0].name.clone();
        let output_name = self.session.outputs[0].name.clone();
        let outputs = self.session.run(vec![(input_name, input_tensor)])?;
        let mask_data = Self::extract_mask_from_outputs_static(&outputs, &self.model_type, &output_name)?;
        Self::create_mask_image_static(&mask_data, self.input_width, self.input_height)
    }
    
    fn apply_mask_efficiently(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>, mask: &ImageBuffer<image::Luma<u8>, Vec<u8>>) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let (width, height) = image.dimensions();
        let (mask_width, mask_height) = mask.dimensions();
        let mut result = ImageBuffer::new(width, height);
        
        for (x, y, pixel) in result.enumerate_pixels_mut() {
            let rgb_pixel = image.get_pixel(x, y);
            let mask_x = (x * mask_width / width).min(mask_width - 1);
            let mask_y = (y * mask_height / height).min(mask_height - 1);
            let alpha = mask.get_pixel(mask_x, mask_y)[0];
            *pixel = image::Rgba([rgb_pixel[0], rgb_pixel[1], rgb_pixel[2], alpha]);
        }
        
        Ok(result)
    }

    pub fn apply_mask_efficiently_public(image: &ImageBuffer<Rgb<u8>, Vec<u8>>, mask: &ImageBuffer<image::Luma<u8>, Vec<u8>>) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let (width, height) = image.dimensions();
        let (mask_width, mask_height) = mask.dimensions();
        
        // Strategy 1: Direct buffer manipulation for maximum speed
        Self::apply_mask_direct_buffer_simd(image, mask, width, height, mask_width, mask_height)
    }
    
    /// Ultra-fast mask application using direct buffer access and SIMD-friendly operations
    fn apply_mask_direct_buffer_simd(
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        mask: &ImageBuffer<image::Luma<u8>, Vec<u8>>,
        width: u32,
        height: u32,
        mask_width: u32,
        mask_height: u32,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let mask_width_usize = mask_width as usize;
        let mask_height_usize = mask_height as usize;
        
        // Pre-allocate result buffer with exact capacity
        let mut result_data = Vec::with_capacity(width_usize * height_usize * 4);
        
        // Get raw data slices for maximum performance
        let image_data = image.as_raw();
        let mask_data = mask.as_raw();
        
        // Pre-calculate scaling factors to avoid repeated division
        let x_scale = mask_width_usize as f32 / width as f32;
        let y_scale = mask_height_usize as f32 / height as f32;
        
        // Strategy 2: Row-by-row processing with cache-friendly access patterns
        for y in 0..height_usize {
            // Calculate mask Y coordinate once per row
            let mask_y = ((y as f32 * y_scale) as usize).min(mask_height_usize - 1);
            let mask_row_offset = mask_y * mask_width_usize;
            
            // Process entire row in chunks for better cache locality
            for x in 0..width_usize {
                let image_idx = (y * width_usize + x) * 3;
                
                // Calculate mask X coordinate with pre-computed scale
                let mask_x = ((x as f32 * x_scale) as usize).min(mask_width_usize - 1);
                let alpha = mask_data[mask_row_offset + mask_x];
                
                // Direct buffer write (SIMD-friendly when compiler optimizes)
                result_data.push(image_data[image_idx]);     // R
                result_data.push(image_data[image_idx + 1]); // G  
                result_data.push(image_data[image_idx + 2]); // B
                result_data.push(alpha);                     // A
            }
        }
        
        // Create ImageBuffer from raw data (zero-copy)
        ImageBuffer::from_raw(width, height, result_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create result image buffer"))
    }
    
    /// Extremely fast mask application using unsafe direct memory access (when safety allows)
    #[allow(dead_code)]
    fn apply_mask_unsafe_simd(
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        mask: &ImageBuffer<image::Luma<u8>, Vec<u8>>,
        width: u32,
        height: u32,
        mask_width: u32,
        mask_height: u32,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let total_pixels = width_usize * height_usize;
        
        // Pre-allocate with uninitialized memory for maximum speed
        let mut result_data = Vec::with_capacity(total_pixels * 4);
        unsafe { result_data.set_len(total_pixels * 4); }
        
        let image_data = image.as_raw();
        let mask_data = mask.as_raw();
        
        // Pre-calculate all mask coordinates (memory vs computation tradeoff)
        let x_scale = mask_width as f32 / width as f32;
        let y_scale = mask_height as f32 / height as f32;
        
        // Vectorized processing in chunks of 4 pixels (SIMD-friendly)
        let chunk_size = 4;
        let full_chunks = total_pixels / chunk_size;
        
        for chunk_idx in 0..full_chunks {
            let base_pixel = chunk_idx * chunk_size;
            
            for i in 0..chunk_size {
                let pixel_idx = base_pixel + i;
                let y = pixel_idx / width_usize;
                let x = pixel_idx % width_usize;
                
                let mask_x = ((x as f32 * x_scale) as usize).min(mask_width as usize - 1);
                let mask_y = ((y as f32 * y_scale) as usize).min(mask_height as usize - 1);
                let alpha = mask_data[mask_y * mask_width as usize + mask_x];
                
                let src_idx = pixel_idx * 3;
                let dst_idx = pixel_idx * 4;
                
                unsafe {
                    *result_data.get_unchecked_mut(dst_idx) = *image_data.get_unchecked(src_idx);
                    *result_data.get_unchecked_mut(dst_idx + 1) = *image_data.get_unchecked(src_idx + 1);
                    *result_data.get_unchecked_mut(dst_idx + 2) = *image_data.get_unchecked(src_idx + 2);
                    *result_data.get_unchecked_mut(dst_idx + 3) = alpha;
                }
            }
        }
        
        // Handle remaining pixels (if total_pixels not divisible by chunk_size)
        for pixel_idx in (full_chunks * chunk_size)..total_pixels {
            let y = pixel_idx / width_usize;
            let x = pixel_idx % width_usize;
            
            let mask_x = ((x as f32 * x_scale) as usize).min(mask_width as usize - 1);
            let mask_y = ((y as f32 * y_scale) as usize).min(mask_height as usize - 1);
            let alpha = mask_data[mask_y * mask_width as usize + mask_x];
            
            let src_idx = pixel_idx * 3;
            let dst_idx = pixel_idx * 4;
            
            unsafe {
                *result_data.get_unchecked_mut(dst_idx) = *image_data.get_unchecked(src_idx);
                *result_data.get_unchecked_mut(dst_idx + 1) = *image_data.get_unchecked(src_idx + 1);
                *result_data.get_unchecked_mut(dst_idx + 2) = *image_data.get_unchecked(src_idx + 2);
                *result_data.get_unchecked_mut(dst_idx + 3) = alpha;
            }
        }
        
        ImageBuffer::from_raw(width, height, result_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create result image buffer"))
    }
    
    fn create_gpu_session(model_path: &Path) -> Result<Session> {
        let providers = [
            ort::execution_providers::cuda::CUDAExecutionProvider::default().build(),
            //ort::execution_providers::tensorrt::TensorRTExecutionProvider::default().build(),
            //ort::execution_providers::rocm::ROCmExecutionProvider::default().build(),
            // TODO turn on/off enabled things later; for now we know our HW is nvidia
        ];
        Session::builder()?
            .with_execution_providers(providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get() / 2)?
            .with_inter_threads(num_cpus::get() / 2)?
            .with_parallel_execution(true)?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")
    }

    fn create_cpu_session(model_path: &Path) -> Result<Session> {
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get() / 2)?
            .with_inter_threads(num_cpus::get() / 2)?
            .with_parallel_execution(true)?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")
    }
    

    fn preprocess_image_efficient(&mut self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<()> {
        self.input_buffer.clear();
        
        let use_imagenet = matches!(self.model_type, ModelType::SINet | ModelType::U2Net | ModelType::YoloV8nSeg | ModelType::FastSam);
        
        for c in 0..3 {
            for y in 0..self.input_height {
                for x in 0..self.input_width {
                    let pixel = image.get_pixel(x as u32, y as u32);
                    let value = pixel[c] as f32 / 255.0;
                    
                    let normalized = if use_imagenet {
                        match c {
                            0 => (value - 0.485) / 0.229,
                            1 => (value - 0.456) / 0.224,
                            2 => (value - 0.406) / 0.225,
                            _ => unreachable!(),
                        }
                    } else {
                        value
                    };
                    
                    self.input_buffer.push(normalized);
                }
            }
        }
        
        Ok(())
    }

    fn extract_mask_from_outputs_static(outputs: &SessionOutputs, _model_type: &ModelType, default_output_name: &str) -> Result<Vec<f32>> {
        if let Some(tensor) = outputs.get(default_output_name) {
            let (_, mask_slice) = tensor.try_extract_tensor::<f32>()?;
            return Ok(mask_slice.to_vec());
        }
        
        if let Some(tensor) = outputs.values().next() {
            let (_, mask_slice) = tensor.try_extract_tensor::<f32>()?;
            return Ok(mask_slice.to_vec());
        }
        
        anyhow::bail!("No output tensor found")
    }
    
    fn create_mask_image_static(mask_data: &[f32], input_width: usize, input_height: usize) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        let output_size = mask_data.len();
        let expected_size = input_height * input_width;
        
        let (mask_width, mask_height, mask_values) = if output_size == expected_size {
            (input_width, input_height, mask_data)
        } else {
            let sqrt_size = (output_size as f64).sqrt() as usize;
            if sqrt_size * sqrt_size == output_size {
                (sqrt_size, sqrt_size, mask_data)
            } else {
                (input_width, input_height, &mask_data[..expected_size.min(output_size)])
            }
        };
        
        let mask_u8: Vec<u8> = mask_values.iter()
            .map(|&x| (x.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        
        ImageBuffer::from_raw(mask_width as u32, mask_height as u32, mask_u8)
            .context("Failed to create mask image")
    }
    
    /// Erode (clip inwards) a mask using Gaussian blur + threshold approach
    /// This creates smooth, natural-looking erosion while maintaining performance
    pub fn erode_mask(mask: &ImageBuffer<image::Luma<u8>, Vec<u8>>, erosion_pixels: u8) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        if erosion_pixels == 0 {
            return Ok(mask.clone());
        }

        let (width, height) = mask.dimensions();
        let width_usize = width as usize;
        let height_usize = height as usize;

        // Convert erosion pixels to sigma for Gaussian blur
        // Higher sigma = more blur = more erosion
        let sigma = (erosion_pixels as f32) * 0.8;

        // Create Gaussian kernel
        let kernel_size = (erosion_pixels as i32 * 2 + 1).max(5) as usize;
        let kernel = Self::create_gaussian_kernel(kernel_size, sigma);

        // Apply Gaussian blur
        let blurred = Self::apply_gaussian_blur(mask, &kernel, kernel_size)?;

        // Threshold the blurred mask to create erosion effect
        // Pixels that were partially transparent after blur become fully transparent
        //let threshold = 255 - (erosion_pixels as u32 * 40).min(200) as u8; // Adaptive threshold
        let threshold = 210;

        let mut eroded_data = Vec::with_capacity(width_usize * height_usize);
        for &pixel in blurred.as_raw() {
            eroded_data.push(if pixel >= threshold { pixel } else { 0 });
        }

        ImageBuffer::from_raw(width, height, eroded_data)
            .context("Failed to create eroded mask")
    }

    /// Create a Gaussian kernel for blur operations
    fn create_gaussian_kernel(size: usize, sigma: f32) -> Vec<f32> {
        let mut kernel = Vec::with_capacity(size * size);
        let center = (size / 2) as i32;
        let sigma_sq = sigma * sigma;
        let two_sigma_sq = 2.0 * sigma_sq;
        let inv_sqrt_two_pi_sigma = 1.0 / (sigma * (2.0 * std::f32::consts::PI).sqrt());

        for y in 0..size {
            for x in 0..size {
                let dx = (x as i32 - center) as f32;
                let dy = (y as i32 - center) as f32;
                let distance_sq = dx * dx + dy * dy;
                let value = inv_sqrt_two_pi_sigma * (-distance_sq / two_sigma_sq).exp();
                kernel.push(value);
            }
        }

        // Normalize kernel
        let sum: f32 = kernel.iter().sum();
        if sum > 0.0 {
            for value in &mut kernel {
                *value /= sum;
            }
        }

        kernel
    }

    /// Apply Gaussian blur to a mask using the given kernel
    fn apply_gaussian_blur(
        mask: &ImageBuffer<image::Luma<u8>, Vec<u8>>,
        kernel: &[f32],
        kernel_size: usize,
    ) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        let (width, height) = mask.dimensions();
        let width_usize = width as usize;
        let height_usize = height as usize;
        let kernel_center = (kernel_size / 2) as i32;

        let mut result_data = Vec::with_capacity(width_usize * height_usize);

        for y in 0..height_usize {
            for x in 0..width_usize {
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let sample_x = x as i32 + kx as i32 - kernel_center;
                        let sample_y = y as i32 + ky as i32 - kernel_center;

                        if sample_x >= 0 && sample_x < width as i32 && sample_y >= 0 && sample_y < height as i32 {
                            let mask_value = mask.get_pixel(sample_x as u32, sample_y as u32)[0] as f32;
                            let kernel_value = kernel[ky * kernel_size + kx];
                            sum += mask_value * kernel_value;
                            weight_sum += kernel_value;
                        }
                    }
                }

                let blurred_value = if weight_sum > 0.0 {
                    (sum / weight_sum).round() as u8
                } else {
                    0
                };

                result_data.push(blurred_value);
            }
        }

        ImageBuffer::from_raw(width, height, result_data)
            .context("Failed to create blurred mask")
    }

}

pub async fn download_model_if_needed(hf_token: Option<String>, disable_download: bool, model_type: Option<ModelType>) -> Result<std::path::PathBuf> {
    let model_type = model_type.unwrap_or(ModelType::U2Net); // Default to working model
    download_model_by_type(model_type, hf_token, disable_download).await
}

pub async fn download_model_by_type(model_type: ModelType, hf_token: Option<String>, disable_download: bool) -> Result<std::path::PathBuf> {
    let cache_dir = get_cache_dir()?;
    let model_path = cache_dir.join(model_type.filename());
    
    if model_path.exists() {
        return Ok(model_path);
    }
    
    std::fs::create_dir_all(&cache_dir)?;
    
    if disable_download {
        return Ok(model_path);
    }
    
    for (_, url) in get_model_download_urls(model_type.clone()) {
        if download_file(&url, &model_path, hf_token.as_deref()).await.is_ok() {
            return Ok(model_path);
        }
        let _ = std::fs::remove_file(&model_path);
    }
    
    Ok(model_path)
}

fn get_cache_dir() -> Result<std::path::PathBuf> {
    let home = std::env::var("HOME")
        .context("HOME environment variable not set")?;
    
    let cache_dir = std::path::Path::new(&home)
        .join(".cache")
        .join("face-overlay-data");
    
    Ok(cache_dir)
}

fn get_model_download_urls(model_type: ModelType) -> Vec<(&'static str, &'static str)> {
    match model_type {
        ModelType::YoloV8nSeg => vec![
            // Pre-converted ONNX models from community sources
            ("Hugging Face (onnx-models)", "https://huggingface.co/onnx-models/yolov8n-seg/resolve/main/yolov8n-seg.onnx"),
            ("GitHub (PrinceB7)", "https://github.com/PrinceB7/yolov8-seg-onnx/releases/download/v1.0/yolov8n-seg.onnx"),
            ("Hugging Face (ultralytics-onnx)", "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/yolov8n-seg.onnx"),
            // Fallback: Use existing U2-Net if YOLOv8 not available
            ("GitHub (rembg)", "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"),
        ],
        ModelType::U2Net => vec![
            // Original U2-Net models
            ("GitHub (rembg)", "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"),
            ("Hugging Face (tomjackson2023)", "https://huggingface.co/tomjackson2023/rembg/resolve/main/u2net.onnx"),
            ("Hugging Face (BritishWerewolf)", "https://huggingface.co/BritishWerewolf/U-2-Net-Human-Seg/resolve/main/u2net.onnx"),
            ("Hugging Face (reidn3r)", "https://huggingface.co/reidn3r/u2net-image-rembg/resolve/main/u2net.onnx"),
            ("GitHub (u2netp)", "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"),
        ],
        ModelType::FastSam => vec![
            // FastSAM models
            ("Ultralytics GitHub", "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt"),
            ("Hugging Face (Ultralytics)", "https://huggingface.co/Ultralytics/FastSAM/resolve/main/FastSAM-s.pt"),
        ],
        ModelType::MediaPipeSelfie => vec![
            // MediaPipe Selfie Segmentation models
            ("Hugging Face (onnx-community)", "https://huggingface.co/onnx-community/mediapipe_selfie_segmentation/resolve/main/general.onnx"),
            ("Hugging Face (qualcomm)", "https://huggingface.co/qualcomm/MediaPipe-Selfie-Segmentation/resolve/main/MediaPipe-Selfie-Segmentation.onnx"),
            ("Backup (onnx-models)", "https://github.com/onnx/models/raw/main/vision/body_analysis/selfie_segmentation/model/selfie_multiclass_256x256.onnx"),
        ],
        ModelType::SINet => vec![
            // SINet Portrait Segmentation models
            ("GitHub (anilsathyan7)", "https://github.com/anilsathyan7/Portrait-Segmentation/releases/download/v1.0/SINet.onnx"),
            ("Portrait-Segmentation", "https://raw.githubusercontent.com/anilsathyan7/Portrait-Segmentation/main/models/SINet.onnx"),
            ("Fast-Portrait-Segmentation", "https://github.com/YexingWan/Fast-Portrait-Segmentation/raw/main/model/SINet.onnx"),
        ],
    }
}

async fn download_file(url: &str, destination: &std::path::Path, hf_token: Option<&str>) -> Result<()> {
    use std::io::Write;
    
    let client = reqwest::Client::new();
    let mut request = client.get(url);
    
    if let Some(token) = hf_token {
        if url.contains("huggingface.co") {
            request = request.header("Authorization", format!("Bearer {}", token));
        }
    }
    
    let response = request.send().await?;
    if !response.status().is_success() {
        anyhow::bail!("Download failed with status: {}", response.status());
    }
    
    let temp_path = destination.with_extension("tmp");
    let mut file = std::fs::File::create(&temp_path)?;
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        file.write_all(&chunk?)?;
    }
    
    file.flush()?;
    drop(file);
    std::fs::rename(&temp_path, destination)?;
    Ok(())
}

pub fn read_hf_token_from_file(token_file: &str) -> Result<String> {
    let token = std::fs::read_to_string(token_file)?.trim().to_string();
    if token.is_empty() {
        anyhow::bail!("Hugging Face token file is empty: {}", token_file);
    }
    Ok(token)
}
