use anyhow::{Context, Result};
use futures_util::StreamExt;
use image::{ImageBuffer, Rgb, Rgba};
use std::path::Path;
use std::collections::VecDeque;

// use ndarray::Array;
use ort::{
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::Value,
};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;

#[cfg(feature = "rocm")]
use ort::execution_providers::ROCmExecutionProvider;

pub fn detect_ai_accelerators() -> Result<Vec<String>> {
    let mut available_accelerators = Vec::new();
    
    tracing::info!("🔍 Detecting AI accelerator hardware...");
    
    // Test CUDA (NVIDIA GPUs)
    if test_cuda_availability() {
        available_accelerators.push("CUDA (NVIDIA GPU)".to_string());
        tracing::info!("✅ CUDA (NVIDIA GPU) - Available");
    } else {
        tracing::info!("❌ CUDA (NVIDIA GPU) - Not available");
    }
    
    // Test TensorRT (NVIDIA GPUs with TensorRT)
    if test_tensorrt_availability() {
        available_accelerators.push("TensorRT (NVIDIA GPU)".to_string());
        tracing::info!("✅ TensorRT (NVIDIA GPU) - Available");
    } else {
        tracing::info!("❌ TensorRT (NVIDIA GPU) - Not available");
    }
    
    // Test ROCm (AMD GPUs)
    if test_rocm_availability() {
        available_accelerators.push("ROCm (AMD GPU)".to_string());
        tracing::info!("✅ ROCm (AMD GPU) - Available");
    } else {
        tracing::info!("❌ ROCm (AMD GPU) - Not available");
    }
    
    // CPU is always available
    available_accelerators.push("CPU".to_string());
    tracing::info!("✅ CPU - Available");
    
    if available_accelerators.len() == 1 {
        tracing::warn!("⚠️  No GPU acceleration available - using CPU only");
    } else {
        tracing::info!("🚀 Found {} AI accelerator(s): {}", 
                      available_accelerators.len(), 
                      available_accelerators.join(", "));
    }
    
    Ok(available_accelerators)
}

fn test_cuda_availability() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Actually test if CUDA is available by trying to create a session
        match Session::builder()
            .and_then(|builder| builder.with_execution_providers([CUDAExecutionProvider::default().build()]))
            .and_then(|builder| builder.commit_from_memory(&[0u8; 32])) // Minimal test data
        {
            Ok(_) => {
                tracing::debug!("CUDA runtime test: Available");
                true
            }
            Err(e) => {
                tracing::debug!("CUDA runtime test failed: {}", e);
                false
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

fn test_tensorrt_availability() -> bool {
    #[cfg(feature = "tensorrt")]
    {
        // Actually test if TensorRT is available by trying to create a session
        match Session::builder()
            .and_then(|builder| builder.with_execution_providers([TensorRTExecutionProvider::default().build()]))
            .and_then(|builder| builder.commit_from_memory(&[0u8; 32])) // Minimal test data
        {
            Ok(_) => {
                tracing::debug!("TensorRT runtime test: Available");
                true
            }
            Err(e) => {
                tracing::debug!("TensorRT runtime test failed: {}", e);
                false
            }
        }
    }
    #[cfg(not(feature = "tensorrt"))]
    {
        false
    }
}

fn test_rocm_availability() -> bool {
    #[cfg(feature = "rocm")]
    {
        // Actually test if ROCm is available by trying to create a session
        match Session::builder()
            .and_then(|builder| builder.with_execution_providers([ROCmExecutionProvider::default().build()]))
            .and_then(|builder| builder.commit_from_memory(&[0u8; 32])) // Minimal test data
        {
            Ok(_) => {
                tracing::debug!("ROCm runtime test: Available");
                true
            }
            Err(e) => {
                tracing::debug!("ROCm runtime test failed: {}", e);
                false
            }
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        false
    }
}

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
    
    pub fn requires_preprocessing(&self) -> bool {
        match self {
            ModelType::U2Net => true,
            ModelType::YoloV8nSeg => true,
            ModelType::FastSam => true,
            ModelType::MediaPipeSelfie => true,  // Normalized input required
            ModelType::SINet => true,  // Normalized input required
        }
    }
}

pub struct MaskCache {
    masks: VecDeque<ImageBuffer<image::Luma<u8>, Vec<u8>>>,
    max_size: usize,
    target_width: u32,
    target_height: u32,
}

impl MaskCache {
    pub fn new(max_size: usize, target_width: u32, target_height: u32) -> Self {
        Self {
            masks: VecDeque::with_capacity(max_size),
            max_size,
            target_width,
            target_height,
        }
    }
    
    pub fn target_dimensions(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }
    
    pub fn add_mask(&mut self, mask: ImageBuffer<image::Luma<u8>, Vec<u8>>) {
        if self.masks.len() >= self.max_size {
            self.masks.pop_front();
        }
        
        // Masks should already be at target resolution
        debug_assert_eq!(mask.dimensions(), (self.target_width, self.target_height), 
                        "Mask dimensions should match target dimensions");
        
        self.masks.push_back(mask);
    }
    
    pub fn get_interpolated_mask(&self) -> Option<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        if self.masks.is_empty() {
            return None;
        }
        
        if self.masks.len() == 1 {
            return Some(self.masks[0].clone());
        }
        
        // Simple averaging of recent masks for temporal coherence
        let mut result: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::new(self.target_width, self.target_height);
        let count = self.masks.len() as f32;
        
        for (x, y, pixel) in result.enumerate_pixels_mut() {
            let mut sum = 0.0;
            for mask in &self.masks {
                sum += mask.get_pixel(x, y)[0] as f32;
            }
            pixel[0] = (sum / count) as u8;
        }
        
        Some(result)
    }
}

pub struct SegmentationModel {
    session: Session,
    input_height: usize,
    input_width: usize,
    model_type: ModelType,
    mask_cache: MaskCache,
    frame_skip_counter: u32,
    ai_inference_interval: u32, // Process AI every N frames
}

impl SegmentationModel {
    pub fn new(model_path: &Path, target_width: u32, target_height: u32) -> Result<Self> {
        Self::new_with_options(model_path, target_width, target_height, ModelType::MediaPipeSelfie, 3, false, None)
    }
    
    pub fn new_with_options(model_path: &Path, target_width: u32, target_height: u32, model_type: ModelType, ai_inference_interval: u32, force_cpu: bool, preferred_provider: Option<&str>) -> Result<Self> {
        tracing::info!("Loading AI segmentation model from: {:?} (type: {:?})", model_path, model_type);
        
        if !model_path.exists() {
            anyhow::bail!("Model file does not exist at path: {:?}", model_path);
        }
        
        // Try GPU acceleration first, fall back to CPU
        let session = Self::create_optimized_session(model_path, force_cpu, preferred_provider)?;
        
        // Debug: Print model input/output info
        tracing::debug!("Model inputs: {:?}", session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        tracing::debug!("Model outputs: {:?}", session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>());

        // Get input dimensions based on model type
        let (input_width, input_height) = model_type.input_size();

        tracing::info!("AI model loaded successfully. Input size: {}x{}, AI inference every {} frames", 
                      input_width, input_height, ai_inference_interval);

        Ok(Self {
            session,
            input_height,
            input_width,
            model_type,
            mask_cache: MaskCache::new(5, target_width, target_height), // Cache last 5 masks
            frame_skip_counter: 0,
            ai_inference_interval,
        })
    }

    pub fn segment_foreground(
        &mut self,
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let (orig_width, orig_height) = image.dimensions();
        
        // Scale input image to target resolution for performance
        let (target_width, target_height) = self.mask_cache.target_dimensions();
        let scaled_image = if (orig_width, orig_height) != (target_width, target_height) {
            image::imageops::resize(
                image,
                target_width,
                target_height,
                image::imageops::FilterType::Nearest, // Fast resize for performance
            )
        } else {
            image.clone()
        };
        
        // Increment frame counter
        self.frame_skip_counter += 1;
        
        // Decide whether to run AI inference or use cached mask
        let mask = if self.frame_skip_counter % self.ai_inference_interval == 0 {
            // Run AI inference on scaled image
            let start_time = std::time::Instant::now();
            let new_mask = self.run_ai_inference(&scaled_image)?;
            let inference_time = start_time.elapsed();
            
            // Cache the new mask (already at target resolution)
            self.mask_cache.add_mask(new_mask.clone());
            
            tracing::debug!("AI inference completed in {:.2}ms", inference_time.as_secs_f64() * 1000.0);
            new_mask
        } else {
            // Use cached mask with interpolation
            match self.mask_cache.get_interpolated_mask() {
                Some(cached_mask) => {
                    tracing::debug!("Using cached mask (frame skip: {})", self.frame_skip_counter % self.ai_inference_interval);
                    cached_mask
                }
                None => {
                    // No cached mask available, run inference
                    tracing::debug!("No cached mask available, running inference");
                    let new_mask = self.run_ai_inference(&scaled_image)?;
                    self.mask_cache.add_mask(new_mask.clone());
                    new_mask
                }
            }
        };

        // Apply mask to scaled image and then scale result back to original if needed
        let scaled_segmented = self.apply_cached_mask(&scaled_image, &mask)?;
        
        let final_result = if (orig_width, orig_height) != (target_width, target_height) {
            image::imageops::resize(
                &scaled_segmented,
                orig_width,
                orig_height,
                image::imageops::FilterType::Nearest, // Fast resize for performance
            )
        } else {
            scaled_segmented
        };
        
        tracing::debug!("Segmentation completed for {}x{} image", orig_width, orig_height);
        
        Ok(final_result)
    }
    
    fn run_ai_inference(&mut self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        // Fast resize with nearest neighbor for better performance
        let resized = image::imageops::resize(
            image,
            self.input_width as u32,
            self.input_height as u32,
            image::imageops::FilterType::Nearest, // Much faster than Lanczos3
        );

        // Optimized preprocessing
        let input_data = self.preprocess_image_fast(&resized)?;
        
        // Create input tensor
        let input_tensor = Value::from_array(([1, 3, self.input_height, self.input_width], input_data))?;
        
        // Run inference and extract mask data in a single scope
        let mask_data = {
            let input_name = self.session.inputs[0].name.clone();
            let output_name = self.session.outputs[0].name.clone();
            let model_type = self.model_type.clone();
            
            let outputs = self.session.run(vec![(input_name, input_tensor)])
                .context("Failed to run model inference")?;
            
            // Extract mask - use appropriate output based on model type
            Self::extract_mask_from_outputs_static(&outputs, &model_type, &output_name)?
        };
        
        // Convert to grayscale mask at AI model resolution
        let mask_image = Self::create_mask_image_static(&mask_data, self.input_width, self.input_height)?;
        
        // Resize mask to target resolution for consistency
        let (target_width, target_height) = self.mask_cache.target_dimensions();
        let final_mask = if (mask_image.width(), mask_image.height()) != (target_width, target_height) {
            image::imageops::resize(
                &mask_image,
                target_width,
                target_height,
                image::imageops::FilterType::Nearest, // Fast resize for performance
            )
        } else {
            mask_image
        };
        
        Ok(final_mask)
    }
    
    fn create_optimized_session(model_path: &Path, force_cpu: bool, preferred_provider: Option<&str>) -> Result<Session> {
        if force_cpu {
            tracing::info!("🖥️  Forced CPU execution (GPU acceleration disabled)");
            return Self::create_cpu_session(model_path);
        }
        
        // Try specific provider if requested
        if let Some(provider) = preferred_provider {
            match provider.to_lowercase().as_str() {
                "cpu" => {
                    tracing::info!("🖥️  Preferred CPU execution");
                    return Self::create_cpu_session(model_path);
                }
                "cuda" => {
                    if let Ok(session) = Self::try_cuda_session(model_path) {
                        return Ok(session);
                    }
                }
                "tensorrt" => {
                    if let Ok(session) = Self::try_tensorrt_session(model_path) {
                        return Ok(session);
                    }
                }
                "rocm" => {
                    if let Ok(session) = Self::try_rocm_session(model_path) {
                        return Ok(session);
                    }
                }
                _ => {
                    tracing::warn!("Unknown GPU provider '{}', trying auto-detection", provider);
                }
            }
        }
        
        // Auto-detect GPU providers in order of preference for Nvidia GPUs
        
        // 1. Try TensorRT first (NVIDIA GPUs with TensorRT) - fastest for inference with lowest latency
        if test_tensorrt_availability() {
            if let Ok(session) = Self::try_tensorrt_session(model_path) {
                return Ok(session);
            }
        }
        
        // 2. Try CUDA (NVIDIA GPUs) - most common, good performance  
        if test_cuda_availability() {
            if let Ok(session) = Self::try_cuda_session(model_path) {
                return Ok(session);
            }
        }
        
        // 3. Try ROCm (AMD GPUs)
        if test_rocm_availability() {
            if let Ok(session) = Self::try_rocm_session(model_path) {
                return Ok(session);
            }
        }
        
        // 4. Fall back to CPU with optimizations
        tracing::info!("🖥️  Using CPU execution (no GPU acceleration available)");
        Self::create_cpu_session(model_path)
    }
    
    #[cfg(feature = "cuda")]
    fn try_cuda_session(model_path: &Path) -> Result<Session> {
        match Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_device_id(0)  // Use primary GPU
                    .build()
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?  // Maximum optimization
            .with_intra_threads(1)?  // Single thread for GPU to avoid overhead
            .commit_from_file(model_path) {
            Ok(session) => {
                tracing::info!("🚀 GPU acceleration enabled: CUDA (low-latency optimized)");
                Ok(session)
            }
            Err(e) => {
                tracing::warn!("CUDA provider failed: {}", e);
                Err(e.into())
            }
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    fn try_cuda_session(_model_path: &Path) -> Result<Session> {
        Err(anyhow::anyhow!("CUDA support not compiled"))
    }
    
    #[cfg(feature = "tensorrt")]
    fn try_tensorrt_session(model_path: &Path) -> Result<Session> {
        match Session::builder()?
            .with_execution_providers([
                TensorRTExecutionProvider::default()
                    .with_device_id(0)  // Use primary GPU
                    .build()
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?  // Maximum optimization
            .with_intra_threads(1)?  // Single thread for GPU to avoid overhead
            .commit_from_file(model_path) {
            Ok(session) => {
                tracing::info!("🚀 GPU acceleration enabled: TensorRT (ultra low-latency optimized)");
                Ok(session)
            }
            Err(e) => {
                tracing::warn!("TensorRT provider failed: {}", e);
                Err(e.into())
            }
        }
    }
    
    #[cfg(not(feature = "tensorrt"))]
    fn try_tensorrt_session(_model_path: &Path) -> Result<Session> {
        Err(anyhow::anyhow!("TensorRT support not compiled"))
    }
    
    #[cfg(feature = "rocm")]
    fn try_rocm_session(model_path: &Path) -> Result<Session> {
        match Session::builder()?
            .with_execution_providers([ROCmExecutionProvider::default().build()])?
            .commit_from_file(model_path) {
            Ok(session) => {
                tracing::info!("🚀 GPU acceleration enabled: ROCm (AMD)");
                Ok(session)
            }
            Err(e) => {
                tracing::warn!("ROCm provider failed: {}", e);
                Err(e.into())
            }
        }
    }
    
    #[cfg(not(feature = "rocm"))]
    fn try_rocm_session(_model_path: &Path) -> Result<Session> {
        Err(anyhow::anyhow!("ROCm support not compiled"))
    }
    
    fn create_cpu_session(model_path: &Path) -> Result<Session> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model - make sure the model file is valid")?;
        
        Ok(session)
    }
    
    fn preprocess_image_fast(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Vec<f32>> {
        let mut input_data = Vec::with_capacity(3 * self.input_height * self.input_width);
        
        // Optimized preprocessing with batch operations
        match self.model_type {
            ModelType::YoloV8nSeg => {
                // YOLOv8 uses 0-1 normalization, no ImageNet normalization
                for c in 0..3 {
                    for y in 0..self.input_height {
                        for x in 0..self.input_width {
                            let pixel = image.get_pixel(x as u32, y as u32);
                            let value = pixel[c] as f32 / 255.0;
                            input_data.push(value);
                        }
                    }
                }
            }
            ModelType::MediaPipeSelfie => {
                // MediaPipe Selfie uses 0-1 normalization (RGB input, not BGR)
                for c in 0..3 {
                    for y in 0..self.input_height {
                        for x in 0..self.input_width {
                            let pixel = image.get_pixel(x as u32, y as u32);
                            let value = pixel[c] as f32 / 255.0;
                            input_data.push(value);
                        }
                    }
                }
            }
            ModelType::SINet => {
                // SINet uses ImageNet normalization but may have specific requirements
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
                            
                            input_data.push(normalized);
                        }
                    }
                }
            }
            _ => {
                // Default normalization (ImageNet)
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
                            
                            input_data.push(normalized);
                        }
                    }
                }
            }
        }
        
        Ok(input_data)
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
    
    fn apply_cached_mask(
        &self,
        original_image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        mask: &ImageBuffer<image::Luma<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let (img_width, img_height) = original_image.dimensions();
        let (mask_width, mask_height) = mask.dimensions();
        
        // Resize mask to match image if needed
        let resized_mask = if (mask_width, mask_height) != (img_width, img_height) {
            image::imageops::resize(
                mask,
                img_width,
                img_height,
                image::imageops::FilterType::Nearest, // Fast resize
            )
        } else {
            mask.clone()
        };
        
        // Apply mask with optimized pixel operations
        let mut result = ImageBuffer::new(img_width, img_height);
        
        for (x, y, pixel) in original_image.enumerate_pixels() {
            let mask_pixel = resized_mask.get_pixel(x, y);
            let alpha = mask_pixel[0]; // Use mask value as alpha
            
            result.put_pixel(
                x,
                y,
                Rgba([pixel[0], pixel[1], pixel[2], alpha]),
            );
        }
        
        Ok(result)
    }

    pub fn input_dimensions(&self) -> (usize, usize) {
        (self.input_width, self.input_height)
    }
    
    pub fn get_ai_inference_interval(&self) -> u32 {
        self.ai_inference_interval
    }
    
    pub fn get_model_type(&self) -> &ModelType {
        &self.model_type
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