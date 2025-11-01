
#![allow(unused_variables, unused_mut, dead_code)]

mod cli;
mod webcam;
mod segmentation;
mod wayland_overlay;

use image::Rgb;
use anyhow::{Context, Result};
use cli::Args;
use image::ImageBuffer;
use segmentation::{download_model_if_needed, read_hf_token_from_file, SegmentationModel, ModelType};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info};
use std::collections::VecDeque;
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
    if args.flip {
        info!("Camera input will be flipped horizontally (mirror effect)");
    }

    let result = run_application(args).await;

    if let Err(e) = result {
        error!("Application error: {:#}", e);
        std::process::exit(1);
    }

    Ok(())
}

fn spawn_webcam_capture_task(
    device: String,
    width: u32,
    height: u32,
    flip: bool,
    tx_camera_frame: tokio::sync::watch::Sender<Option<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
    cancel_bool: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    tokio::spawn(async move {
        let mut webcam = WebcamCapture::new(Some(&device), width, height, flip)
            .context("Failed to initialize webcam")?;
        let mut allowed_errors = 10;
        while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
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
    })
}

fn spawn_frame_slicing_task(
    mut rx_camera_frame: tokio::sync::watch::Receiver<Option<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
    mut rx_ai_mask_data: tokio::sync::watch::Receiver<Option<ImageBuffer<image::Luma<u8>, Vec<u8>>>>,
    tx_sliced_frame: tokio::sync::watch::Sender<Option<ImageBuffer<image::Rgba<u8>, Vec<u8>>>>,
    border_width: u32,
    border_color: [u8; 3],
    cancel_bool: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    tokio::spawn(async move {
        let mut allowed_errors = 10;
        while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
            // Grab the latest from the webcam
            let frame_value = (*rx_camera_frame.borrow()).clone();
            rx_camera_frame.mark_unchanged(); // Say we have observed the current value

            // Grab the latest AI mask
            let ai_mask = (*rx_ai_mask_data.borrow()).clone();
            rx_ai_mask_data.mark_unchanged(); // Say we have observed the current value

            if let (Some(frame), Some(mask)) = (frame_value, ai_mask) {
                // Slice it and send to the screen! (this should be FAST)
                match SegmentationModel::apply_mask_with_border(&frame, &mask, border_width, border_color) {
                    Ok(frame_a) => {
                        // send
                        if let Err(e) = tx_sliced_frame.send(Some(frame_a)) {
                            eprintln!("[ tx_sliced_frame.send ] {:?}", e);
                        }
                    }
                    Err(e) => {
                        error!("[ SegmentationModel::apply_mask_with_border ] {:?}", e);
                    }
                }
            }
        }
        Ok(())
    })
}

fn spawn_ai_mask_generation_task(
    mut rx_camera_frame: tokio::sync::watch::Receiver<Option<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
    segmentation_model: Option<SegmentationModel>,
    tx_ai_mask_data: tokio::sync::watch::Sender<Option<ImageBuffer<image::Luma<u8>, Vec<u8>>>>,
    mask_erosion: u8,
    cancel_bool: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    tokio::spawn(async move {
        let mut segmentation_model = segmentation_model.expect("We have no segmentation_model!");
        let mut allowed_errors = 10;
        while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
            // Grab the latest from the webcam
            let frame_value = (*rx_camera_frame.borrow()).clone();
            rx_camera_frame.mark_unchanged(); // Say we have observed the current value
            if let Some(frame) = frame_value {
                // Slice image and update meta-data for slice_task
                let resized_image = image::imageops::resize(&frame, segmentation_model.input_width as u32, segmentation_model.input_height as u32, image::imageops::FilterType::Nearest);

                let mut mask = segmentation_model.run_efficient_ai_inference(&resized_image)?;
                
                // Apply mask erosion if requested
                if mask_erosion > 0 {
                    mask = SegmentationModel::erode_mask(&mask, mask_erosion)?;
                }

                // send slice data; this is async to the entire imaging pipeline
                if let Err(e) = tx_ai_mask_data.send(Some(mask)) {
                   eprintln!("[ tx_ai_mask_data.send ] {:?}", e);
                }
            }
        }
        Ok(())
    })
}

fn spawn_zoom_input_reader(
    cancel_bool: std::sync::Arc<std::sync::atomic::AtomicBool>,
    input_event_file: String,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    use evdev::Device;
    tokio::task::spawn_blocking(move || {
        let mut allowed_errors = 10;
        while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
            match Device::open(&input_event_file) {
                Ok(mut device) => {
                    if let Err(e) = device.set_nonblocking(true) {
                        eprintln!("{}:{} {:?}", file!(), line!(), e);
                    }
                    while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
                        match device.fetch_events() {
                            Ok(event_stream) => {
                                for event in event_stream {
                                    eprintln!("{}:{} {:?}", file!(), line!(), event);
                                    if cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                if e.kind() == std::io::ErrorKind::WouldBlock {
                                    // This is fine, continue to next loop iteration.
                                }
                                else {
                                    allowed_errors -= 1;
                                    eprintln!("{}:{} {:?}", file!(), line!(), e);
                                }
                                if cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
                                    break;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    allowed_errors -= 1;
                    eprintln!("{}:{} {:?}", file!(), line!(), e);
                }
            }
        }
        Ok(())
    })
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

    let anchor_position: AnchorPosition = args.anchor.clone().into();
    let (overlay_width, overlay_height) = {
        let webcam = WebcamCapture::new(Some(&args.device), args.width, args.height, args.flip)
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

    // FPS tracking for rolling average of last 20 frames
    let mut frame_times: VecDeque<Instant> = VecDeque::with_capacity(20);
    let mut frame_count = 0u64;
    let mut last_fps_report = Instant::now();

    // Write "true" to signal that threads should exit
    let thread_cancel_bool = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Only retains the most-recent copy of the data
    let (tx_camera_frame, mut rx_camera_frame) = tokio::sync::watch::channel::<Option< ImageBuffer<Rgb<u8>, Vec<u8>> >>(None);

    // Spawn an infinite loop which reads frames from the camera and pushed them into the channel
    let webcam_thread_cancel_bool = thread_cancel_bool.clone();
    let webcam_task = spawn_webcam_capture_task(
        args.device.clone(),
        args.width,
        args.height,
        args.flip,
        tx_camera_frame,
        webcam_thread_cancel_bool,
    );

    // The slicer_task needs to see this data
    let (tx_ai_mask_data, mut rx_ai_mask_data) = tokio::sync::watch::channel::<Option< ImageBuffer<image::Luma<u8>, Vec<u8>> >>(None);

    // Spawn an infinite loop which reads frames from the camera and pushed them into the channel
    let (tx_sliced_frame, mut rx_sliced_frame) = tokio::sync::watch::channel::<Option< ImageBuffer<image::Rgba<u8>, Vec<u8>> >>(None);
    let slicer_thread_cancel_bool = thread_cancel_bool.clone();
    let slicer_rx_camera_frame = rx_camera_frame.clone();
    let slicer_task = spawn_frame_slicing_task(
        slicer_rx_camera_frame,
        rx_ai_mask_data.clone(),
        tx_sliced_frame,
        args.border_width,
        args.get_border_color(),
        slicer_thread_cancel_bool,
    );

    // Finally, spawn an infinite AI loop which maintains the pixel mask data to slice against
    let ai_mask_thread_cancel_bool = thread_cancel_bool.clone();
    let ai_mask_rx_camera_frame = rx_camera_frame.clone();
    let ai_mask_task = spawn_ai_mask_generation_task(
        ai_mask_rx_camera_frame,
        segmentation_model,
        tx_ai_mask_data,
        args.mask_erosion,
        ai_mask_thread_cancel_bool,
    );

    let zoom_input_reader_cancel_bool = thread_cancel_bool.clone();
    let zoom_input_reader_task = spawn_zoom_input_reader(
        zoom_input_reader_cancel_bool,
        args.input_events_file
    );

    let mut allowed_errors = 10;
    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(last_frame_time);

        if elapsed >= frame_duration {
            // Track frame timing for FPS calculation
            frame_times.push_back(now);
            frame_count += 1;

            // Keep only last 20 frame times
            if frame_times.len() > 20 {
                frame_times.pop_front();
            }

            // Calculate and report FPS every second
            if last_fps_report.elapsed() >= Duration::from_secs(1) {
                let rolling_fps = if frame_times.len() >= 2 {
                    let time_span = frame_times.back().unwrap().duration_since(*frame_times.front().unwrap());
                    if time_span.as_millis() > 0 {
                        (frame_times.len() - 1) as f64 / time_span.as_secs_f64()
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let shm_stats = overlay.get_shm_stats();
                info!("FPS: {:.1} (rolling avg of last {} frames) | SHM: {} mapping, {}% buffer utilization", 
                      rolling_fps, frame_times.len(),
                      if shm_stats.persistent_mapping_active { "persistent" } else { "standard" },
                      shm_stats.buffer_utilization);
                last_fps_report = now;
            }

            let frame_value = (*rx_sliced_frame.borrow()).clone();
            rx_sliced_frame.mark_unchanged(); // Say we have observed the current value
            if let Some(frame) = frame_value {
                if let Err(e) = process_frame(&frame, &mut overlay, &qh).await {
                    error!("Frame processing error: {}", e);
                    allowed_errors -= 1;
                    if allowed_errors < 1 {
                        break;
                    }
                }
            }

            last_frame_time = now;
        }

        // Async event handling, none of these lines block
        for _ in 0..8 {
            match event_queue.dispatch_pending(&mut overlay).context("Failed to dispatch Wayland events") {
                Ok(_evts) => { }
                Err(e) => {
                    eprintln!("[ event_queue.dispatch_pending ] {:?}", e);
                }
            }
            if let Some(guard) = event_queue.prepare_read() {
                if let Err(e) = guard.read() { // read from socket
                    if !format!("{:?}", e).contains("WouldBlock") {
                        eprintln!("[ event_queue.prepare_read ] {:?}", e);
                    }
                }
            }
            if let Err(e) = event_queue.flush() {
                eprintln!("[ event_queue.flush ] {:?}", e);
            }
        }

        if webcam_task.is_finished() || slicer_task.is_finished() || ai_mask_task.is_finished() {
            break;
        }
    }


    thread_cancel_bool.store(true, std::sync::atomic::Ordering::Relaxed);
    sleep(Duration::from_millis(110)).await;

    if !webcam_task.is_finished() {
        webcam_task.abort();
    }
    if !slicer_task.is_finished() {
        slicer_task.abort();
    }
    if !ai_mask_task.is_finished() {
        ai_mask_task.abort();
    }

    Ok(())
}

async fn process_frame(
    processed_frame: &ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    overlay: &mut WaylandOverlay,
    qh: &wayland_client::QueueHandle<WaylandOverlay>,
) -> Result<()> {

    // Automatically reposition overlay periodically to avoid mouse interference
    overlay.auto_reposition(qh);

    overlay.update_frame(processed_frame, qh).context("Failed to update overlay frame")
}
