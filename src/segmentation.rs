use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb, Rgba};
use std::path::Path;
use tracing::warn;

pub struct SegmentationModel {
    _placeholder: (),
}

impl SegmentationModel {
    pub fn new(_model_path: &Path) -> Result<Self> {
        warn!("AI segmentation is disabled (ort dependency not available)");
        anyhow::bail!("AI segmentation not compiled in this build")
    }

    pub fn segment_foreground(
        &self,
        _image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        anyhow::bail!("AI segmentation not compiled in this build")
    }

}

pub fn download_model_if_needed() -> Result<std::path::PathBuf> {
    let model_dir = std::path::Path::new("models");
    let model_path = model_dir.join("u2net.onnx");
    
    if !model_path.exists() {
        warn!("Model not found at {:?}. You need to download a segmentation model.", model_path);
        warn!("You can download U2-Net ONNX model from:");
        warn!("https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface");
        warn!("Or convert a PyTorch model to ONNX format.");
        
        std::fs::create_dir_all(model_dir)
            .context("Failed to create models directory")?;
        
        anyhow::bail!("Model file not found. Please download and place the model at: {:?}", model_path);
    }
    
    Ok(model_path)
}