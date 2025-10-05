use face_overlay::segmentation::detect_ai_accelerators;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();
        
    println!("Testing AI accelerator detection...");
    
    match detect_ai_accelerators() {
        Ok(accelerators) => {
            println!("Detection completed. Found {} accelerators.", accelerators.len());
            for (i, acc) in accelerators.iter().enumerate() {
                println!("  {}. {}", i + 1, acc);
            }
        }
        Err(e) => {
            println!("Detection failed: {}", e);
        }
    }
}
