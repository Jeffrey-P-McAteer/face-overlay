mod cli;
mod webcam;
mod segmentation;
mod wayland_overlay;
mod mouse_tracker;

use image::Rgb;
use anyhow::{Context, Result};
use cli::Args;
use image::ImageBuffer;
use segmentation::{download_model_if_needed, read_hf_token_from_file, SegmentationModel, ModelType};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info};
use wayland_overlay::{AnchorPosition, WaylandOverlay};
use webcam::WebcamCapture;

fn load_model(path: &std::path::Path, model_type: ModelType) -> Option<SegmentationModel> {
    if path.exists() {
        if let Ok(model) = SegmentationModel::new(path, model_type.clone()) {
            return Some(model);
        }
    }
    
    let u2net_path = std::env::var("HOME")
        .map(|home| std::path::PathBuf::from(format!("{}/.cache/face-overlay-data/u2net.onnx", home)))
        .unwrap_or_default();
    
    if u2net_path.exists() && model_type != ModelType::U2Net {
        SegmentationModel::new(&u2net_path, ModelType::U2Net).ok()
    } else {
        None
    }
}


#[tokio::main(flavor = "multi_thread", worker_threads = 6)]
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
    let hf_token = args.hf_token_file.as_ref()
        .and_then(|f| read_hf_token_from_file(f).ok());
    
    let model_path = if let Some(_) = args.model_path {
        args.get_model_path()
    } else {
        download_model_if_needed(hf_token, args.disable_download, Some(args.get_ai_model_type()))
            .await.unwrap_or_default()
    };

    let mut segmentation_model = load_model(&model_path, args.get_ai_model_type());

    let anchor_position: AnchorPosition = args.anchor.into();
    let (overlay_width, overlay_height) = {
        let webcam = WebcamCapture::new(Some(&args.device), args.width, args.height)
            .context("Failed to initialize webcam")?; // Camera briefly open+closed
        webcam.resolution()
    };
    let (mut overlay, _conn, mut event_queue) = WaylandOverlay::new(anchor_position, overlay_width, overlay_height)
        .context("Failed to initialize Wayland overlay")?;

    let qh = event_queue.handle();
    overlay.create_layer_surface(&qh)
        .context("Failed to create layer surface")?;


    let frame_duration = Duration::from_millis(1000 / args.fps as u64);
    let mut last_frame_time = Instant::now();

    // Only retains the most-recent copy of the data
    let (tx_camera_frame, mut rx_camera_frame) = tokio::sync::watch::channel::<Option< ImageBuffer<Rgb<u8>, Vec<u8>> >>(None);

    // Spawn an infinite loop which reads frames from the camera and pushed them into the channel
    let webcam_task: tokio::task::JoinHandle<Result<(), anyhow::Error>> = tokio::spawn(async move {
        let mut webcam = WebcamCapture::new(Some(&args.device), args.width, args.height)
            .context("Failed to initialize webcam")?;
        let mut allowed_errors = 10;
        while allowed_errors > 0 {
            match webcam.capture_frame().context("Failed to capture webcam frame") {
                Ok(frame) => {
                    if let Err(e) = tx_camera_frame.send(Some(frame)) {
                        eprintln!("{:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("{:?}", e);
                    allowed_errors -= 1;
                }
            }
        }
        Ok(())
    });

    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(last_frame_time);

        if elapsed >= frame_duration {
            // if let Err(e) = process_frame(&mut webcam, &mut segmentation_model, &mut overlay, &qh).await {
            //     error!("Frame processing error: {}", e);
            //     break;
            // }

            let frame_value = (*rx_camera_frame.borrow()).clone();
            rx_camera_frame.mark_unchanged(); // Say we have observed the current value
            if let Some(frame) = frame_value {
                if let Err(e) = process_frame(&frame, &mut segmentation_model, &mut overlay, &qh).await {
                    error!("Frame processing error: {}", e);
                    break;
                }
                info!("Processed frame!");
            }


            last_frame_time = now;
        }

        // Async event handling, none of these lines block
        match event_queue.dispatch_pending(&mut overlay).context("Failed to dispatch Wayland events") {
            Ok(_evts) => { }
            Err(e) => {
                eprintln!("[ event_queue.dispatch_pending ] {:?}", e);
            }
        }
        if let Some(guard) = event_queue.prepare_read() {
            if let Err(e) = guard.read() { // read from socket
                eprintln!("[ event_queue.prepare_read ] {:?}", e);
            }
        }
        if let Err(e) = event_queue.flush() {
            eprintln!("[ event_queue.flush ] {:?}", e);
        }

        let sleep_time = frame_duration.saturating_sub(now.elapsed());
        if sleep_time > Duration::from_millis(1) {
            sleep(sleep_time).await;
        }

        if webcam_task.is_finished() {
            break;
        }
    }

    Ok(())
}

async fn process_frame(
    frame: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    segmentation_model: &mut Option<SegmentationModel>,
    overlay: &mut WaylandOverlay,
    qh: &wayland_client::QueueHandle<WaylandOverlay>,
) -> Result<()> {
    let (width, height) = frame.dimensions();

    let processed_frame = match segmentation_model {
        Some(model) => model.segment_foreground(frame).unwrap_or_else(|_| {
            ImageBuffer::from_fn(width, height, |x, y| {
                let pixel = frame.get_pixel(x, y);
                image::Rgba([pixel[0], pixel[1], pixel[2], 255])
            })
        }),
        None => ImageBuffer::from_fn(width, height, |x, y| {
            let pixel = frame.get_pixel(x, y);
            image::Rgba([pixel[0], pixel[1], pixel[2], 255])
        }),
    };

    // for i in 0..100 {
    //     processed_frame.put_pixel(i, i, image::Rgba([255u8, 255u8, 255u8, 255u8]) );
    // }

    // Automatically reposition overlay periodically to avoid mouse interference
    overlay.auto_reposition(qh);

    overlay.update_frame(&processed_frame, qh).context("Failed to update overlay frame")
}
