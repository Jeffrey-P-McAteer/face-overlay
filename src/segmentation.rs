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
    input_height: usize,
    input_width: usize,
    model_type: ModelType,
    // Pre-allocated buffer for AI data reuse and maximum efficiency
    input_buffer: Vec<f32>,
}

impl SegmentationModel {
    pub fn new(model_path: &Path, model_type: ModelType) -> Result<Self> {
        tracing::info!("Loading CPU-optimized AI segmentation model from: {:?} (type: {:?})", model_path, model_type);
        
        if !model_path.exists() {
            anyhow::bail!("Model file does not exist at path: {:?}", model_path);
        }
        
        // Create CPU-only session
        let session = Self::create_cpu_session(model_path)?;
        
        // Debug: Print model input/output info
        tracing::debug!("Model inputs: {:?}", session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        tracing::debug!("Model outputs: {:?}", session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>());

        // Get input dimensions based on model type
        let (input_width, input_height) = model_type.input_size();

        tracing::info!("🖥️ CPU-optimized AI model loaded successfully. Input size: {}x{}", 
                      input_width, input_height);

        // Pre-allocate buffer for maximum efficiency
        let buffer_size = 3 * input_height * input_width;

        Ok(Self {
            session,
            input_height,
            input_width,
            model_type,
            input_buffer: Vec::with_capacity(buffer_size),
        })
    }

    pub fn segment_foreground(
        &mut self,
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let start_time = std::time::Instant::now();
        
        // Resize input to AI model size (MediaPipe Selfie: 256x256 for maximum speed)
        let resized_image = image::imageops::resize(
            image,
            self.input_width as u32,
            self.input_height as u32,
            image::imageops::FilterType::Nearest, // Fastest resize method
        );

        // Run AI inference on EVERY frame for maximum responsiveness
        let mask = match self.run_efficient_ai_inference(&resized_image) {
            Ok(mask) => mask,
            Err(e) => {
                tracing::error!("⚠️  AI inference failed: {}", e);
                std::thread::sleep(std::time::Duration::from_millis(250));
                
                // Return fully transparent image on AI failure
                let (width, height) = image.dimensions();
                return Ok(ImageBuffer::from_fn(width, height, |_, _| image::Rgba([0, 0, 0, 0])));
            }
        };

        // Apply mask directly to original image for best quality
        let segmented = self.apply_mask_efficiently(image, &mask)?;
        
        let inference_time = start_time.elapsed();
        tracing::debug!("🚀 AI segmentation completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);
        
        Ok(segmented)
    }
    
    fn run_efficient_ai_inference(&mut self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        // Ultra-fast preprocessing with pre-allocated buffer reuse
        self.preprocess_image_efficient(image)?;
        
        // Create input tensor with buffer reuse for maximum efficiency
        let input_tensor = Value::from_array(([1, 3, self.input_height, self.input_width], self.input_buffer.clone()))?;
        
        // Run inference with minimal allocations
        let mask_data = {
            let input_name = self.session.inputs[0].name.clone();
            let output_name = self.session.outputs[0].name.clone();
            let model_type = self.model_type.clone();
            
            let outputs = self.session.run(vec![(input_name, input_tensor)])
                .context("🚨 AI model inference failed")?;
            
            // Extract mask efficiently
            Self::extract_mask_from_outputs_static(&outputs, &model_type, &output_name)?
        };
        
        // Convert to mask image at model resolution (no unnecessary resizing)
        let mask_image = Self::create_mask_image_static(&mask_data, self.input_width, self.input_height)?;
        
        Ok(mask_image)
    }
    
    fn apply_mask_efficiently(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>, mask: &ImageBuffer<image::Luma<u8>, Vec<u8>>) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let (width, height) = image.dimensions();
        let (mask_width, mask_height) = mask.dimensions();
        
        // Create output buffer efficiently
        let mut result = ImageBuffer::new(width, height);
        
        // Apply mask with efficient pixel processing
        for (x, y, pixel) in result.enumerate_pixels_mut() {
            let rgb_pixel = image.get_pixel(x, y);
            
            // Scale mask coordinates efficiently
            let mask_x = (x * mask_width / width).min(mask_width - 1);
            let mask_y = (y * mask_height / height).min(mask_height - 1);
            let alpha = mask.get_pixel(mask_x, mask_y)[0];
            
            *pixel = image::Rgba([rgb_pixel[0], rgb_pixel[1], rgb_pixel[2], alpha]);
        }
        
        Ok(result)
    }
    
    
    fn create_cpu_session(model_path: &Path) -> Result<Session> {
        tracing::info!("🖥️ Creating CPU-optimized ONNX session with {} threads", num_cpus::get());
        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model - make sure the model file is valid")?;
        
        Ok(session)
    }
    

    fn preprocess_image_efficient(&mut self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<()> {
        // Clear and reuse the pre-allocated buffer for maximum efficiency
        self.input_buffer.clear();
        
        // Optimized preprocessing specifically for MediaPipe Selfie (the default fast model)
        match self.model_type {
            ModelType::MediaPipeSelfie => {
                // MediaPipe Selfie uses 0-1 normalization (RGB input, not BGR)
                // Ultra-optimized loop with direct buffer writing
                for c in 0..3 {
                    for y in 0..self.input_height {
                        for x in 0..self.input_width {
                            let pixel = image.get_pixel(x as u32, y as u32);
                            let value = pixel[c] as f32 / 255.0;
                            self.input_buffer.push(value);
                        }
                    }
                }
            }
            ModelType::SINet => {
                // SINet uses ImageNet normalization
                for c in 0..3 {
                    for y in 0..self.input_height {
                        for x in 0..self.input_width {
                            let pixel = image.get_pixel(x as u32, y as u32);
                            let value = pixel[c] as f32 / 255.0;
                            
                            let normalized = match c {
                                0 => (value - 0.485) / 0.229, // Red channel
                                1 => (value - 0.456) / 0.224, // Green channel  
                                2 => (value - 0.406) / 0.225, // Blue channel
                                _ => unreachable!(),
                            };
                            
                            self.input_buffer.push(normalized);
                        }
                    }
                }
            }
            _ => {
                // Default normalization for other models
                for c in 0..3 {
                    for y in 0..self.input_height {
                        for x in 0..self.input_width {
                            let pixel = image.get_pixel(x as u32, y as u32);
                            let value = pixel[c] as f32 / 255.0;
                            
                            let normalized = match c {
                                0 => (value - 0.485) / 0.229, // Red channel
                                1 => (value - 0.456) / 0.224, // Green channel  
                                2 => (value - 0.406) / 0.225, // Blue channel
                                _ => unreachable!(),
                            };
                            
                            self.input_buffer.push(normalized);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    fn extract_mask_from_outputs_static(outputs: &SessionOutputs, model_type: &ModelType, default_output_name: &str) -> Result<Vec<f32>> {
        match model_type {
            ModelType::YoloV8nSeg => {
                // YOLOv8 outputs multiple tensors, typically we need the mask output
                // Look for outputs that might contain masks
                for (name, tensor) in outputs.iter() {
                    if name.contains("mask") || name.contains("output") || outputs.len() == 1 {
                        let (_, mask_slice) = tensor.try_extract_tensor::<f32>()?;
                        return Ok(mask_slice.to_vec());
                    }
                }
                // Fallback: use first output
                let first_tensor = outputs.values().next().unwrap();
                let (_, mask_slice) = first_tensor.try_extract_tensor::<f32>()?;
                Ok(mask_slice.to_vec())
            }
            ModelType::MediaPipeSelfie => {
                // MediaPipe Selfie outputs a single mask tensor (256x256x1)
                let first_tensor = outputs.values().next()
                    .context("No output tensor found for MediaPipe Selfie")?;
                let (_, mask_slice) = first_tensor.try_extract_tensor::<f32>()?;
                Ok(mask_slice.to_vec())
            }
            ModelType::SINet => {
                // SINet outputs a single mask tensor 
                let first_tensor = outputs.values().next()
                    .context("No output tensor found for SINet")?;
                let (_, mask_slice) = first_tensor.try_extract_tensor::<f32>()?;
                Ok(mask_slice.to_vec())
            }
            _ => {
                // Default: use first output (U2-Net style)
                let (_, mask_slice) = outputs.get(default_output_name)
                    .context("Output tensor not found")?
                    .try_extract_tensor::<f32>()?;
                Ok(mask_slice.to_vec())
            }
        }
    }
    
    fn create_mask_image_static(mask_data: &[f32], input_width: usize, input_height: usize) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        let output_size = mask_data.len();
        let expected_size = input_height * input_width;
        
        // Handle different output shapes
        let (mask_width, mask_height, mask_values) = if output_size == expected_size {
            (input_width, input_height, mask_data.to_vec())
        } else {
            // Try to infer dimensions
            let sqrt_size = (output_size as f64).sqrt() as usize;
            if sqrt_size * sqrt_size == output_size {
                (sqrt_size, sqrt_size, mask_data.to_vec())
            } else {
                // Fallback: resize to expected dimensions
                tracing::warn!("Unexpected mask output size: {}, expected: {}", output_size, expected_size);
                (input_width, input_height, 
                 mask_data.into_iter().take(expected_size).cloned().collect())
            }
        };
        
        // Convert to u8 mask
        let mask_u8: Vec<u8> = mask_values
            .into_iter()
            .map(|x| (x.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        
        ImageBuffer::from_raw(mask_width as u32, mask_height as u32, mask_u8)
            .context("Failed to create mask image")
    }
    
}

pub async fn download_model_if_needed(hf_token: Option<String>, disable_download: bool, model_type: Option<ModelType>) -> Result<std::path::PathBuf> {
    let model_type = model_type.unwrap_or(ModelType::U2Net); // Default to working model
    download_model_by_type(model_type, hf_token, disable_download).await
}

pub async fn download_model_by_type(model_type: ModelType, hf_token: Option<String>, disable_download: bool) -> Result<std::path::PathBuf> {
    let cache_dir = get_cache_dir()?;
    let model_filename = model_type.filename();
    let model_path = cache_dir.join(model_filename);
    
    if model_path.exists() {
        let metadata = std::fs::metadata(&model_path)?;
        let size_mb = metadata.len() / 1024 / 1024;
        tracing::info!("Found existing AI model at {:?} ({} MB, type: {:?})", model_path, size_mb, model_type);
        return Ok(model_path);
    }
    
    std::fs::create_dir_all(&cache_dir)
        .context("Failed to create cache directory")?;
    
    if disable_download {
        tracing::debug!("AI model not found at {:?}, automatic download disabled", model_path);
        tracing::debug!("To enable AI segmentation, place a {} model file at {:?}", model_filename, model_path);
        return Ok(model_path);
    }
    
    tracing::info!("AI model not found, attempting automatic download for {:?}...", model_type);
    
    // Try downloading from multiple sources
    let download_urls = get_model_download_urls(model_type.clone());
    let mut last_error = None;
    
    for (source, url) in download_urls {
        tracing::info!("Attempting download from {} ({})", source, url);
        
        match download_file(&url, &model_path, hf_token.as_deref()).await {
            Ok(_) => {
                let metadata = std::fs::metadata(&model_path)?;
                let size_mb = metadata.len() / 1024 / 1024;
                tracing::info!("Successfully downloaded AI model from {} ({} MB, type: {:?})", source, size_mb, model_type);
                return Ok(model_path);
            }
            Err(e) => {
                tracing::warn!("Failed to download from {}: {}", source, e);
                last_error = Some(e);
                // Clean up partial download
                let _ = std::fs::remove_file(&model_path);
            }
        }
    }
    
    if let Some(error) = last_error {
        tracing::warn!("All download attempts failed. Last error: {}", error);
        tracing::info!("To enable AI segmentation, manually place a {} model file at {:?}", model_filename, model_path);
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
    
    tracing::debug!("Starting download from: {}", url);
    
    let client = reqwest::Client::new();
    let mut request = client.get(url);
    
    // Add Hugging Face authorization header if token is provided
    if let Some(token) = hf_token {
        if url.contains("huggingface.co") {
            request = request.header("Authorization", format!("Bearer {}", token));
            tracing::debug!("Using Hugging Face token for authentication");
        }
    }
    
    let response = request.send().await
        .context("Failed to start download")?;
    
    if !response.status().is_success() {
        anyhow::bail!("Download failed with status: {} - {}", response.status(), response.status().canonical_reason().unwrap_or("Unknown"));
    }
    
    let total_size = response.content_length();
    if let Some(size) = total_size {
        tracing::info!("Download size: {:.1} MB", size as f64 / 1024.0 / 1024.0);
    }
    
    // Create a temporary file first
    let temp_path = destination.with_extension("tmp");
    let mut file = std::fs::File::create(&temp_path)
        .context("Failed to create temporary file")?;
    
    let mut stream = response.bytes_stream();
    let mut downloaded = 0u64;
    let mut last_progress = 0;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Failed to read download chunk")?;
        file.write_all(&chunk)
            .context("Failed to write to file")?;
        
        downloaded += chunk.len() as u64;
        
        // Show progress every 10MB or at completion
        if let Some(total) = total_size {
            let progress = (downloaded * 100 / total) as u32;
            if progress >= last_progress + 10 || downloaded == total {
                tracing::info!("Download progress: {}% ({:.1} MB / {:.1} MB)", 
                    progress, 
                    downloaded as f64 / 1024.0 / 1024.0,
                    total as f64 / 1024.0 / 1024.0
                );
                last_progress = progress;
            }
        }
    }
    
    file.flush().context("Failed to flush file")?;
    drop(file);
    
    // Move temp file to final destination
    std::fs::rename(&temp_path, destination)
        .context("Failed to move downloaded file to final location")?;
    
    tracing::debug!("Download completed successfully");
    Ok(())
}

pub fn read_hf_token_from_file(token_file: &str) -> Result<String> {
    let token = std::fs::read_to_string(token_file)
        .with_context(|| format!("Failed to read Hugging Face token from file: {}", token_file))?;
    
    let token = token.trim();
    if token.is_empty() {
        anyhow::bail!("Hugging Face token file is empty: {}", token_file);
    }
    
    tracing::debug!("Successfully read Hugging Face token from file");
    Ok(token.to_string())
}