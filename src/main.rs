mod cli;
mod webcam;
mod segmentation;
mod wayland_overlay;
mod mouse_tracker;

use anyhow::{Context, Result};
use cli::Args;
use image::ImageBuffer;
use mouse_tracker::MouseEventHandler;
use segmentation::{download_model_if_needed, read_hf_token_from_file, SegmentationModel, ModelType};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use wayland_overlay::{AnchorPosition, WaylandOverlay};
use webcam::WebcamCapture;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse_args();
    args.setup_logging();

    info!("Starting face-overlay application");

    let result = run_application(args).await;

    if let Err(e) = result {
        error!("Application error: {:#}", e);
        std::process::exit(1);
    }

    Ok(())
}

async fn run_application(args: Args) -> Result<()> {
    info!("Initializing face overlay application...");
    
    // CPU-optimized segmentation
    info!("🖥️ Using CPU-optimized AI segmentation");
    
    // Read Hugging Face token if provided
    let hf_token = if let Some(token_file) = &args.hf_token_file {
        match read_hf_token_from_file(token_file) {
            Ok(token) => {
                info!("Successfully loaded Hugging Face token from {}", token_file);
                Some(token)
            }
            Err(e) => {
                warn!("Failed to read Hugging Face token: {}", e);
                None
            }
        }
    } else {
        None
    };
    
    let model_path = if args.model_path.is_some() {
        args.get_model_path()
    } else {
        download_model_if_needed(hf_token, args.disable_download, Some(args.get_ai_model_type())).await.unwrap_or_else(|e| {
            debug!("Model setup failed: {}", e);
            std::path::PathBuf::new()
        })
    };

    let mut webcam = WebcamCapture::new(Some(&args.device), args.width, args.height)
        .context("Failed to initialize webcam")?;
    
    // Report webcam status
    if webcam.is_simulated() {
        info!("Using simulated camera feed (no real webcam available)");
    } else {
        info!("Using real webcam: {}", args.device);
    }

    let mut segmentation_model = if model_path.exists() {
        match SegmentationModel::new(&model_path, args.get_ai_model_type()) {
            Ok(model) => {
                info!("CPU-optimized AI segmentation model loaded successfully");
                Some(model)
            }
            Err(e) => {
                debug!("Failed to load segmentation model: {}", e);
                
                // Try fallback to U2-Net if it exists
                let u2net_path = std::env::var("HOME")
                    .map(|home| std::path::PathBuf::from(format!("{}/.cache/face-overlay-data/u2net.onnx", home)))
                    .unwrap_or_default();
                
                if u2net_path.exists() && args.get_ai_model_type() != ModelType::U2Net {
                    info!("Falling back to U2-Net model");
                    match SegmentationModel::new(&u2net_path, ModelType::U2Net) {
                        Ok(model) => {
                            info!("U2-Net fallback model loaded successfully");
                            Some(model)
                        }
                        Err(e) => {
                            debug!("Failed to load U2-Net fallback model: {}", e);
                            None
                        }
                    }
                } else {
                    None
                }
            }
        }
    } else {
        // Try fallback to U2-Net if it exists
        let u2net_path = std::env::var("HOME")
            .map(|home| std::path::PathBuf::from(format!("{}/.cache/face-overlay-data/u2net.onnx", home)))
            .unwrap_or_default();
        
        if u2net_path.exists() {
            info!("No {} model found, falling back to U2-Net model", args.ai_model);
            match SegmentationModel::new(&u2net_path, ModelType::U2Net) {
                Ok(model) => {
                    info!("U2-Net fallback model loaded successfully");
                    Some(model)
                }
                Err(e) => {
                    debug!("Failed to load U2-Net fallback model: {}", e);
                    None
                }
            }
        } else {
            debug!("No AI segmentation model available, using raw video feed");
            None
        }
    };

    let anchor_position: AnchorPosition = args.anchor.into();
    let (overlay_width, overlay_height) = webcam.resolution();
    let (mut overlay, _conn, mut event_queue) = WaylandOverlay::new(anchor_position, overlay_width, overlay_height)
        .context("Failed to initialize Wayland overlay")?;

    let qh = event_queue.handle();
    overlay.create_layer_surface(&qh)
        .context("Failed to create layer surface")?;

    let mut mouse_handler = MouseEventHandler::new(args.mouse_flip_delay, !args.disable_mouse_flip)
        .unwrap_or_else(|e| {
            warn!("Mouse tracking disabled: {}", e);
            MouseEventHandler::new(args.mouse_flip_delay, false).unwrap()
        });

    let frame_duration = Duration::from_millis(1000 / args.fps as u64);
    let mut last_frame_time = Instant::now();

    info!("Starting main loop with {}fps target", args.fps);

    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(last_frame_time);

        if elapsed >= frame_duration {
            if let Err(e) = process_frame(
                &mut webcam,
                &mut segmentation_model,
                &mut overlay,
                &mut mouse_handler,
                &qh,
            ).await {
                error!("Frame processing error: {}", e);
                break;
            }

            last_frame_time = now;
        }

        event_queue.blocking_dispatch(&mut overlay)
            .context("Failed to dispatch Wayland events")?;

        let sleep_time = frame_duration.saturating_sub(now.elapsed());
        if sleep_time > Duration::from_millis(1) {
            sleep(sleep_time).await;
        }
    }

    info!("Application shutting down");
    Ok(())
}

async fn process_frame(
    webcam: &mut WebcamCapture,
    segmentation_model: &mut Option<SegmentationModel>,
    overlay: &mut WaylandOverlay,
    mouse_handler: &mut MouseEventHandler,
    qh: &wayland_client::QueueHandle<WaylandOverlay>,
) -> Result<()> {
    let frame = webcam.capture_frame()
        .context("Failed to capture webcam frame")?;

    let processed_frame = if let Some(model) = segmentation_model {
        match model.segment_foreground(&frame) {
            Ok(segmented) => {
                debug!("🚀 AI segmentation successful");
                segmented
            },
            Err(e) => {
                error!("🚨 Critical AI segmentation failure: {}", e);
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                warn!("⚠️  Falling back to full visibility mode due to AI failure");
                
                // Efficient fallback: convert to RGBA with full opacity
                let (width, height) = frame.dimensions();
                ImageBuffer::from_fn(width, height, |x, y| {
                    let pixel = frame.get_pixel(x, y);
                    image::Rgba([pixel[0], pixel[1], pixel[2], 255]) // Full opacity
                })
            }
        }
    } else {
        debug!("No AI model available, using raw video feed");
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        
        // Efficient fallback: full opacity
        let (width, height) = frame.dimensions();
        ImageBuffer::from_fn(width, height, |x, y| {
            let pixel = frame.get_pixel(x, y);
            image::Rgba([pixel[0], pixel[1], pixel[2], 255]) // Full opacity
        })
    };

    let mouse_position = overlay.get_mouse_position();
    if mouse_handler.check_for_flip(overlay.get_surface_bounds(), mouse_position) {
        let current_anchor = overlay.get_anchor_position();
        let new_anchor = match current_anchor {
            AnchorPosition::LowerLeft => AnchorPosition::LowerRight,
            AnchorPosition::LowerRight => AnchorPosition::LowerLeft,
        };
        overlay.set_anchor_position(new_anchor, qh);
        mouse_handler.reset_flip_state();
        info!("🔄 Auto-flipped overlay from {} to {} after 1000ms mouse dwell", 
              match current_anchor {
                  AnchorPosition::LowerLeft => "lower-left",
                  AnchorPosition::LowerRight => "lower-right",
              },
              match new_anchor {
                  AnchorPosition::LowerLeft => "lower-left", 
                  AnchorPosition::LowerRight => "lower-right",
              });
    }

    overlay.update_frame(&processed_frame, &qh)
        .context("Failed to update overlay frame")?;

    Ok(())
}
