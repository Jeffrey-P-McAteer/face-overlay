use anyhow::{Context, Result};
use futures_util::StreamExt;
use image::{ImageBuffer, Rgb, Rgba};
use std::path::Path;

use ndarray::Array;
use ort::{
    session::Session,
    value::Value,
};

pub struct SegmentationModel {
    session: Session,
    input_height: usize,
    input_width: usize,
}

impl SegmentationModel {
    pub fn new(model_path: &Path) -> Result<Self> {
        tracing::info!("Loading AI segmentation model from: {:?}", model_path);
        
        let session = Session::builder()?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model - make sure the model file is valid")?;

        // Get input dimensions from the model - default to 320x320 for U2-Net
        let input_height = 320;
        let input_width = 320;

        tracing::info!("AI model loaded successfully. Input size: {}x{}", input_width, input_height);

        Ok(Self {
            session,
            input_height,
            input_width,
        })
    }

    pub fn segment_foreground(
        &mut self,
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let (orig_width, orig_height) = image.dimensions();
        
        // Resize image to model input size
        let resized = image::imageops::resize(
            image,
            self.input_width as u32,
            self.input_height as u32,
            image::imageops::FilterType::Lanczos3,
        );

        // Preprocess image for model input
        let input_data = self.preprocess_image(&resized)?;
        
        // Create input tensor using the simpler tuple format
        let input_tensor = Value::from_array(([1, 3, self.input_height, self.input_width], input_data))?;
        
        // Run inference and extract mask data
        let mask_data = {
            let outputs = self.session.run(ort::inputs!["input" => input_tensor])
                .context("Failed to run model inference")?;
            
            // Extract mask from output
            let output_tensor = outputs["output"].try_extract_tensor::<f32>()?;
            let (_, mask_slice) = output_tensor;
            
            // Convert slice to ndarray for processing
            Array::from_shape_vec((1, 1, self.input_height, self.input_width), mask_slice.to_vec())?
        };

        // Post-process mask and apply to original image
        let segmented = self.apply_segmentation_mask(image, &mask_data.into_dyn(), orig_width, orig_height)?;
        
        tracing::debug!("Segmentation completed for {}x{} image", orig_width, orig_height);
        
        Ok(segmented)
    }

    fn preprocess_image(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Vec<f32>> {
        let mut input_data = Vec::with_capacity(3 * self.input_height * self.input_width);
        
        // Convert to CHW format (channels, height, width) and normalize
        for c in 0..3 {
            for y in 0..self.input_height {
                for x in 0..self.input_width {
                    let pixel = image.get_pixel(x as u32, y as u32);
                    let value = pixel[c] as f32 / 255.0;
                    
                    // Apply ImageNet normalization
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
        
        Ok(input_data)
    }

    fn apply_segmentation_mask(
        &self,
        original_image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        mask_data: &ndarray::ArrayD<f32>,
        target_width: u32,
        target_height: u32,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        // Resize mask to match target dimensions
        let mask_slice = mask_data.as_slice().context("Failed to get mask slice")?;
        
        // Create binary mask (threshold at 0.5)
        let binary_mask: Vec<u8> = mask_slice
            .iter()
            .map(|&x| if x > 0.5 { 255 } else { 0 })
            .collect();

        let mask_image = ImageBuffer::from_raw(
            self.input_width as u32,
            self.input_height as u32,
            binary_mask,
        )
        .context("Failed to create mask image")?;

        // Resize mask to match original image size
        let resized_mask = image::imageops::resize(
            &mask_image,
            target_width,
            target_height,
            image::imageops::FilterType::Lanczos3,
        );

        // Apply mask to create RGBA image with transparency
        let (width, height) = original_image.dimensions();
        let mut result = ImageBuffer::new(width, height);

        for (x, y, pixel) in original_image.enumerate_pixels() {
            let mask_pixel: &image::Luma<u8> = resized_mask.get_pixel(x, y);
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
}

pub async fn download_model_if_needed(hf_token: Option<String>, disable_download: bool) -> Result<std::path::PathBuf> {
    let cache_dir = get_cache_dir()?;
    let model_path = cache_dir.join("u2net.onnx");
    
    if model_path.exists() {
        let metadata = std::fs::metadata(&model_path)?;
        let size_mb = metadata.len() / 1024 / 1024;
        tracing::info!("Found existing AI model at {:?} ({} MB)", model_path, size_mb);
        return Ok(model_path);
    }
    
    std::fs::create_dir_all(&cache_dir)
        .context("Failed to create cache directory")?;
    
    if disable_download {
        tracing::debug!("AI model not found at {:?}, automatic download disabled", model_path);
        tracing::debug!("To enable AI segmentation, place a u2net.onnx model file at {:?}", model_path);
        return Ok(model_path);
    }
    
    tracing::info!("AI model not found, attempting automatic download...");
    
    // Try downloading from multiple sources
    let download_urls = get_model_download_urls();
    let mut last_error = None;
    
    for (source, url) in download_urls {
        tracing::info!("Attempting download from {} ({})", source, url);
        
        match download_file(&url, &model_path, hf_token.as_deref()).await {
            Ok(_) => {
                let metadata = std::fs::metadata(&model_path)?;
                let size_mb = metadata.len() / 1024 / 1024;
                tracing::info!("Successfully downloaded AI model from {} ({} MB)", source, size_mb);
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
        tracing::info!("To enable AI segmentation, manually place a u2net.onnx model file at {:?}", model_path);
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

fn get_model_download_urls() -> Vec<(&'static str, &'static str)> {
    vec![
        // GitHub releases (fastest, no auth needed)
        ("GitHub (rembg)", "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"),
        
        // Hugging Face direct download URLs (converted from blob to resolve format)
        ("Hugging Face (tomjackson2023)", "https://huggingface.co/tomjackson2023/rembg/resolve/main/u2net.onnx"),
        ("Hugging Face (BritishWerewolf)", "https://huggingface.co/BritishWerewolf/U-2-Net-Human-Seg/resolve/main/u2net.onnx"),
        ("Hugging Face (reidn3r)", "https://huggingface.co/reidn3r/u2net-image-rembg/resolve/main/u2net.onnx"),
        
        // Alternative lightweight model
        ("GitHub (u2netp)", "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"),
    ]
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