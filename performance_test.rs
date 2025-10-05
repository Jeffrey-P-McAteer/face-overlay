use std::time::Instant;
use face_overlay::segmentation::{SegmentationModel, ModelType};
use image::{ImageBuffer, Rgb};

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_level(true)
        .with_target(false)
        .init();

    let model_path = std::env::var("HOME")
        .map(|home| std::path::PathBuf::from(format!("{}/.cache/face-overlay-data/u2net.onnx", home)))
        .unwrap_or_default();

    if !model_path.exists() {
        println!("Model not found at {:?}. Please download it first by running the main application.", model_path);
        return Ok(());
    }

    println!("=== Face Overlay Performance Test ===");
    println!("Testing optimizations with different AI inference intervals...\n");

    // Test different AI inference intervals
    let test_configs = vec![
        (1, "No frame skipping (baseline)"),
        (3, "AI every 3rd frame (default optimization)"),
        (5, "AI every 5th frame (aggressive optimization)"),
        (10, "AI every 10th frame (maximum optimization)"),
    ];

    for (interval, description) in test_configs {
        test_performance_with_interval(interval, description, &model_path)?;
        println!();
    }

    Ok(())
}

fn test_performance_with_interval(
    ai_inference_interval: u32,
    description: &str,
    model_path: &std::path::Path,
) -> anyhow::Result<()> {
    println!("Testing: {} (AI inference interval: {})", description, ai_inference_interval);
    
    // Create segmentation model with optimization settings
    let mut model = SegmentationModel::new_with_options(
        model_path,
        320,
        240,
        ModelType::U2Net,
        ai_inference_interval,
    )?;

    // Create test frames (simulate webcam feed)
    let test_frames = create_test_frames(10);
    
    let start_time = Instant::now();
    let mut total_inference_time = 0.0;
    let mut inference_count = 0;
    
    for (i, frame) in test_frames.iter().enumerate() {
        let frame_start = Instant::now();
        
        match model.segment_foreground(frame) {
            Ok(_) => {
                let frame_time = frame_start.elapsed().as_secs_f64() * 1000.0;
                
                // Count actual AI inferences (every nth frame)
                if (i + 1) % ai_inference_interval as usize == 0 {
                    total_inference_time += frame_time;
                    inference_count += 1;
                }
                
                if i == 0 {
                    println!("  First frame processed in: {:.2}ms", frame_time);
                }
            }
            Err(e) => {
                println!("  Error processing frame {}: {}", i, e);
            }
        }
    }
    
    let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
    let avg_frame_time = total_time / test_frames.len() as f64;
    let theoretical_fps = 1000.0 / avg_frame_time;
    
    println!("  Total time for {} frames: {:.2}ms", test_frames.len(), total_time);
    println!("  Average time per frame: {:.2}ms", avg_frame_time);
    println!("  Theoretical FPS: {:.1}", theoretical_fps);
    println!("  AI inferences performed: {}/{} frames", inference_count, test_frames.len());
    
    if inference_count > 0 {
        let avg_inference_time = total_inference_time / inference_count as f64;
        println!("  Average AI inference time: {:.2}ms", avg_inference_time);
    }
    
    // Check if we're achieving 24+ FPS target
    if theoretical_fps >= 24.0 {
        println!("  ✅ Achieves 24+ FPS target!");
    } else {
        println!("  ⚠️  Below 24 FPS target");
    }
    
    Ok(())
}

fn create_test_frames(count: usize) -> Vec<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let mut frames = Vec::new();
    
    for i in 0..count {
        // Create a simple test pattern - start with smaller dimensions to avoid issues
        let width = 320;
        let height = 240;
        let mut img = ImageBuffer::new(width, height);
        
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let r = ((x + i as u32 * 10) % 256) as u8;
            let g = ((y + i as u32 * 20) % 256) as u8;
            let b = ((x + y + i as u32 * 30) % 256) as u8;
            *pixel = Rgb([r, g, b]);
        }
        
        frames.push(img);
    }
    
    frames
}