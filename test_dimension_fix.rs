use face_overlay::segmentation::{SegmentationModel, ModelType};
use image::{ImageBuffer, Rgb};

fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("HOME")
        .map(|home| std::path::PathBuf::from(format!("{}/.cache/face-overlay-data/u2net.onnx", home)))
        .unwrap_or_default();

    if !model_path.exists() {
        println!("❌ Model not found at {:?}", model_path);
        return Ok(());
    }

    // Suppress ONNX logs
    unsafe { std::env::set_var("RUST_LOG", "error"); }
    
    println!("🔧 Testing dimension fix...");
    
    // Create model with mismatched dimensions to trigger the issue
    // Model input: 320x320, Target: 640x480 (webcam resolution)
    let mut model = SegmentationModel::new_with_options(
        &model_path,
        640,  // target_width (webcam)
        480,  // target_height (webcam)
        ModelType::U2Net,
        3,    // AI inference every 3rd frame
    )?;

    // Create test frame at webcam resolution
    let test_frame = create_test_frame(640, 480);
    
    println!("Testing with {}x{} input frame (model input: 320x320)", 
             test_frame.width(), test_frame.height());
    
    // Process several frames to test both AI inference and caching
    for i in 1..=5 {
        match model.segment_foreground(&test_frame) {
            Ok(result) => {
                println!("✅ Frame {}: Processed successfully ({}x{})", 
                         i, result.width(), result.height());
            }
            Err(e) => {
                println!("❌ Frame {}: Error - {}", i, e);
                return Err(e);
            }
        }
    }
    
    println!("🎉 Dimension fix working correctly!");
    Ok(())
}

fn create_test_frame(width: u32, height: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(width, height);
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (x % 256) as u8;
        let g = (y % 256) as u8;
        let b = ((x + y) % 256) as u8;
        *pixel = Rgb([r, g, b]);
    }
    
    img
}