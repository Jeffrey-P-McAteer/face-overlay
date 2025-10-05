use std::time::Instant;
use face_overlay::segmentation::{SegmentationModel, ModelType};
use image::{ImageBuffer, Rgb};

fn main() -> anyhow::Result<()> {
    // Test if model exists
    let model_path = std::env::var("HOME")
        .map(|home| std::path::PathBuf::from(format!("{}/.cache/face-overlay-data/u2net.onnx", home)))
        .unwrap_or_default();

    if !model_path.exists() {
        println!("❌ Model not found at {:?}", model_path);
        println!("Please download it first by running: ./target/release/face-overlay --help");
        return Ok(());
    }

    println!("🎯 Face Overlay Performance Validation");
    println!("======================================");
    println!("Model: U2-Net (167MB) at {:?}", model_path);
    println!();

    // Test frame skipping optimization
    test_frame_skipping_optimization(&model_path)?;
    
    Ok(())
}

fn test_frame_skipping_optimization(model_path: &std::path::Path) -> anyhow::Result<()> {
    let test_configs = [
        (1, "Baseline (AI every frame)"),
        (3, "Optimized (AI every 3rd frame)"),
        (5, "Aggressive (AI every 5th frame)"),
    ];

    for (interval, description) in test_configs {
        println!("Testing: {}", description);
        
        // Suppress ORT logs by redirecting stderr temporarily
        let mut model = {
            let stderr = std::process::Stdio::null();
            SegmentationModel::new_with_options(
                model_path,
                320,
                240,
                ModelType::U2Net,
                interval,
            )?
        };

        // Create test frame
        let test_frame = create_test_frame();
        
        // Time 5 frames
        let start_time = Instant::now();
        for i in 0..5 {
            let frame_start = Instant::now();
            if let Ok(_) = model.segment_foreground(&test_frame) {
                let frame_time = frame_start.elapsed().as_secs_f64() * 1000.0;
                if i == 0 {
                    println!("  First frame: {:.1}ms", frame_time);
                }
            }
        }
        
        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let avg_frame_time = total_time / 5.0;
        let theoretical_fps = 1000.0 / avg_frame_time;
        
        println!("  Average per frame: {:.1}ms", avg_frame_time);
        println!("  Theoretical FPS: {:.1}", theoretical_fps);
        
        if theoretical_fps >= 24.0 {
            println!("  ✅ Achieves 24+ FPS target!");
        } else {
            println!("  ⚠️  Below 24 FPS target");
        }
        
        println!();
    }
    
    Ok(())
}

fn create_test_frame() -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(320, 240);
    
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (x % 256) as u8;
        let g = (y % 256) as u8;
        let b = ((x + y) % 256) as u8;
        *pixel = Rgb([r, g, b]);
    }
    
    img
}