
#![allow(unused_variables, unused_mut, dead_code)]

use anyhow::{Context, Result};
use image::{ImageBuffer, Rgba};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
        },
        WaylandSurface,
    },
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_buffer, wl_output, wl_shm, wl_surface},
    Connection, QueueHandle,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnchorPosition {
    LowerLeft,
    LowerRight,
}

impl From<AnchorPosition> for Anchor {
    fn from(pos: AnchorPosition) -> Self {
        match pos {
            AnchorPosition::LowerLeft => Anchor::BOTTOM | Anchor::LEFT,
            AnchorPosition::LowerRight => Anchor::BOTTOM | Anchor::RIGHT,
        }
    }
}

pub struct WaylandOverlay {
    registry_state: RegistryState,
    output_state: OutputState,
    compositor_state: CompositorState,
    shm: wayland_client::protocol::wl_shm::WlShm,
    layer_shell: LayerShell,
    
    layer_surface: Option<LayerSurface>,
    pool: Option<wayland_client::protocol::wl_shm_pool::WlShmPool>,
    buffer: Option<wl_buffer::WlBuffer>,
    
    width: u32,
    height: u32,
    anchor_position: AnchorPosition,
    ideal_anchor_position: AnchorPosition,
    
    outputs: HashMap<wl_output::WlOutput, OutputInfo>,
    last_position_change: Instant,
    auto_reposition_interval: Duration,
    mouse_last_known_pos: std::sync::Arc<std::sync::RwLock<(i32, i32)>>,
    mouse_detection_margin: i32,
    
    // Optimized persistent SHM resource management
    current_shm_fd: Option<i32>,
    current_buffer_size: u32,
    persistent_memory_map: Option<*mut u8>,  // Keep memory mapped across frames
    max_buffer_size: u32,  // Track largest buffer size allocated
}

#[derive(Debug, Clone)]
struct OutputInfo {
    width: i32,
    height: i32,
}

impl WaylandOverlay {
    pub fn new(anchor_position: AnchorPosition, width: u32, height: u32) -> Result<(Self, Connection, wayland_client::EventQueue<Self>)> {
        // Check environment variables
        let wayland_display = std::env::var("WAYLAND_DISPLAY").unwrap_or_else(|_| "not set".to_string());
        let xdg_session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "unknown".to_string());
        
        info!("Wayland environment: WAYLAND_DISPLAY={}, XDG_SESSION_TYPE={}", wayland_display, xdg_session_type);
        
        let conn = Connection::connect_to_env()
            .context("Failed to connect to Wayland compositor. Make sure you're running in a Wayland session (Sway, GNOME on Wayland, etc.)")?;

        let (globals, event_queue) = registry_queue_init(&conn)
            .context("Failed to initialize Wayland registry")?;

        let qh = event_queue.handle();

        // Debug: List available protocols
        info!("Available Wayland protocols:");
        let protocol_list = globals.contents().with_list(|list| {
            list.iter().map(|g| (g.interface.clone(), g.version)).collect::<Vec<_>>()
        });
        for (interface, version) in protocol_list {
            info!("  - {} v{}", interface, version);
        }

        let registry_state = RegistryState::new(&globals);
        let output_state = OutputState::new(&globals, &qh);
        let compositor_state = CompositorState::bind(&globals, &qh)
            .context("Compositor not available - this usually means the Wayland compositor doesn't support the required protocols")?;
        
        
        let shm = globals
            .bind::<wayland_client::protocol::wl_shm::WlShm, _, _>(&qh, 1..=1, ())
            .context("Shared memory not available")?;
        
        let layer_shell = LayerShell::bind(&globals, &qh)
            .context("Layer shell not available - this compositor doesn't support wlr-layer-shell protocol. Sway, wlroots-based compositors, and some others support this.")?;

        info!("Successfully connected to Wayland compositor with layer shell support");



        let overlay = Self {
            registry_state,
            output_state,
            compositor_state,
            shm,
            layer_shell,
            layer_surface: None,
            pool: None,
            buffer: None,
            width,
            height,
            anchor_position,
            ideal_anchor_position: anchor_position.clone(),
            outputs: HashMap::new(),
            last_position_change: Instant::now(),
            auto_reposition_interval: Duration::from_secs(2), // Fallback: move every 2 seconds
            mouse_last_known_pos: std::sync::Arc::new(std::sync::RwLock::new( (-1, -1) )),
            mouse_detection_margin: 25, // Move when mouse is within 50 pixels
            current_shm_fd: None,
            current_buffer_size: 0,
            persistent_memory_map: None,
            max_buffer_size: 0,
        };

        Ok((overlay, conn, event_queue))
    }

    pub fn create_layer_surface(&mut self, qh: &QueueHandle<Self>) -> Result<()> {
        info!("Creating layer surface with size {}x{}, anchor: {:?}", self.width, self.height, self.anchor_position);
        
        let surface = self.compositor_state.create_surface(qh);
        
        let layer_surface = self.layer_shell.create_layer_surface(
            qh,
            surface,
            Layer::Overlay,
            Some("face-overlay"),
            None,
        );

        layer_surface.set_anchor(self.anchor_position.into());
        layer_surface.set_exclusive_zone(-1);  // Don't reserve space
        layer_surface.set_margin(0, 0, 0, 0);  // No margins - extend to screen edges
        layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer_surface.set_size(self.width, self.height);

        let surface_clone = layer_surface.wl_surface().clone();
        
        // Create an explicit empty region for true input pass-through
        let compositor = &self.compositor_state.wl_compositor().clone();
        let empty_region = compositor.create_region(qh, ());
        // Don't add any rectangles to the region - leave it empty for full pass-through
        
        // Set the empty input region to ensure ALL mouse/keyboard events pass through
        surface_clone.set_input_region(Some(&empty_region));
        surface_clone.commit();
        
        self.layer_surface = Some(layer_surface);
        
        info!("Layer surface created successfully with anchor: {:?}", self.anchor_position);
        info!("Surface will extend to screen edges in {} corner with no margins", 
              match self.anchor_position {
                  AnchorPosition::LowerLeft => "lower-left",
                  AnchorPosition::LowerRight => "lower-right",
              });
        info!("🖱️  Sway-compatible input pass-through enabled - empty input region ensures events go through overlay");
        
        Ok(())
    }

    pub fn update_frame(&mut self, image: &ImageBuffer<Rgba<u8>, Vec<u8>>, qh: &QueueHandle<Self>) -> Result<()> {
        // Check if layer surface exists (avoid borrowing issues)
        if self.layer_surface.is_none() {
            warn!("No layer surface available for frame update");
            return Ok(());
        }

        let (img_width, img_height) = image.dimensions();
        let stride = img_width * 4; // 4 bytes per pixel (ARGB)
        let buffer_size = stride * img_height;
        
        // Only recreate SHM pool if buffer size actually increased beyond current capacity
        let need_new_pool = self.pool.is_none() || 
            img_width != self.width || 
            img_height != self.height ||
            buffer_size > self.max_buffer_size;
        
        if need_new_pool {
            debug!("Recreating SHM pool: size changed from {}x{} ({} bytes) to {}x{} ({} bytes)", 
                   self.width, self.height, self.max_buffer_size, 
                   img_width, img_height, buffer_size);
            
            // Clean up existing resources first
            self.cleanup_shm_resources();
            
            // Update dimensions and layer surface size
            if img_width != self.width || img_height != self.height {
                self.width = img_width;
                self.height = img_height;
                if let Some(layer_surface) = &self.layer_surface {
                    layer_surface.set_size(self.width, self.height);
                }
                debug!("Layer surface size changed to {}x{}", self.width, self.height);
            }
            
            // Calculate expanded buffer size with 25% headroom to reduce future recreations
            let expanded_buffer_size = ((buffer_size as f32) * 1.25) as u32;
            
            // Create new shared memory buffer with expanded size
            let shm_fd = self.create_shm_fd(expanded_buffer_size as usize)?;
            
            // Create persistent memory mapping for the entire buffer
            let memory_ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    expanded_buffer_size as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    shm_fd,
                    0,
                )
            };
            
            if memory_ptr == libc::MAP_FAILED {
                unsafe { libc::close(shm_fd); }
                anyhow::bail!("Failed to create persistent memory mapping for SHM");
            }
            
            // Create Wayland SHM pool
            use std::os::fd::BorrowedFd;
            let borrowed_fd = unsafe { BorrowedFd::borrow_raw(shm_fd) };
            let pool = self.shm.create_pool(borrowed_fd, expanded_buffer_size as i32, qh, ());
            
            // Store new resources
            self.pool = Some(pool);
            self.current_shm_fd = Some(shm_fd);
            self.current_buffer_size = buffer_size;
            self.max_buffer_size = expanded_buffer_size;
            self.persistent_memory_map = Some(memory_ptr as *mut u8);
            
            info!("Created persistent SHM pool: {}x{} ({} bytes allocated, {} bytes headroom)", 
                  img_width, img_height, buffer_size, expanded_buffer_size - buffer_size);
        } else {
            // Just update current frame size for existing pool
            self.current_buffer_size = buffer_size;
        }
        
        // Create buffer from existing pool
        let pool = self.pool.as_ref().unwrap();
        let buffer = pool.create_buffer(
            0,                           // offset
            self.width as i32,          // width
            self.height as i32,         // height  
            stride as i32,              // stride
            wl_shm::Format::Argb8888,   // format
            qh,
            (),
        );
        
        // Write pixel data to persistent memory mapping (no mmap/munmap overhead)
        if let Some(memory_ptr) = self.persistent_memory_map {
            self.write_pixels_to_persistent_memory(image, memory_ptr, buffer_size as usize)?;
        } else {
            anyhow::bail!("No persistent memory mapping available");
        }
        
        // Attach buffer and commit (get layer_surface reference here)
        if let Some(layer_surface) = &self.layer_surface {
            let surface = layer_surface.wl_surface();
            surface.attach(Some(&buffer), 0, 0);
            surface.damage(0, 0, self.width as i32, self.height as i32);
            surface.commit();
        }
        
        // Clean up old buffer (but keep pool and FD for reuse)
        if let Some(old_buffer) = self.buffer.take() {
            old_buffer.destroy();
        }
        
        // Store new buffer
        self.buffer = Some(buffer);
        
        debug!("Frame updated: {}x{} with {} SHM pool (persistent mapping: {})", 
               self.width, self.height,
               if need_new_pool { "new" } else { "reused" },
               if self.persistent_memory_map.is_some() { "active" } else { "inactive" });
        
        Ok(())
    }

    /// Clean up shared memory resources to prevent file descriptor leaks
    fn cleanup_shm_resources(&mut self) {
        // Unmap persistent memory mapping first
        if let Some(memory_ptr) = self.persistent_memory_map.take() {
            unsafe {
                libc::munmap(memory_ptr as *mut libc::c_void, self.max_buffer_size as usize);
            }
            debug!("Unmapped persistent memory mapping ({} bytes)", self.max_buffer_size);
        }
        
        // Close current file descriptor if exists
        if let Some(fd) = self.current_shm_fd.take() {
            unsafe {
                libc::close(fd);
            }
            debug!("Closed shared memory file descriptor: {}", fd);
        }
        
        // Clean up old Wayland resources
        if let Some(old_buffer) = self.buffer.take() {
            old_buffer.destroy();
        }
        if let Some(old_pool) = self.pool.take() {
            old_pool.destroy();
        }
        
        self.current_buffer_size = 0;
        self.max_buffer_size = 0;
    }

    fn create_shm_fd(&self, size: usize) -> Result<i32> {
        use std::ffi::CString;
        
        // Create anonymous file descriptor using memfd_create
        let name = CString::new("face-overlay-shm").unwrap();
        let fd = unsafe {
            libc::syscall(libc::SYS_memfd_create, name.as_ptr(), libc::MFD_CLOEXEC)
        };
        
        let fd = if fd == -1 {
            // Fallback to creating a temporary file if memfd_create fails
            warn!("memfd_create failed, falling back to temporary file");
            
            // Create a temporary file manually to get better control
            let temp_path = std::env::temp_dir().join(format!("face-overlay-shm-{}", std::process::id()));
            let fd = unsafe {
                let path_cstr = CString::new(temp_path.to_str().unwrap()).unwrap();
                let fd = libc::open(
                    path_cstr.as_ptr(),
                    libc::O_CREAT | libc::O_RDWR | libc::O_EXCL | libc::O_CLOEXEC,
                    0o600
                );
                
                if fd != -1 {
                    // Unlink the file immediately so it gets cleaned up when closed
                    libc::unlink(path_cstr.as_ptr());
                }
                
                fd
            };
            
            if fd == -1 {
                anyhow::bail!("Failed to create temporary file for SHM");
            }
            
            fd
        } else {
            fd as i32
        };
        
        // Truncate to required size
        unsafe {
            if libc::ftruncate(fd, size as i64) == -1 {
                libc::close(fd); // Clean up on error
                anyhow::bail!("Failed to resize SHM file");
            }
        }
        
        debug!("Created SHM file descriptor {} for {} bytes", fd, size);
        Ok(fd)
    }
    
    /// Optimized pixel writing to persistent memory mapping (no mmap/munmap overhead)
    fn write_pixels_to_persistent_memory(&self, image: &ImageBuffer<Rgba<u8>, Vec<u8>>, memory_ptr: *mut u8, size: usize) -> Result<()> {
        let (img_width, img_height) = image.dimensions();
        let buffer = unsafe { std::slice::from_raw_parts_mut(memory_ptr, size) };
        let image_data = image.as_raw();
        
        // Optimized pixel conversion with direct memory access and SIMD-friendly loops
        let total_pixels = (img_width * img_height) as usize;
        
        // Process pixels in chunks for better cache performance
        for pixel_idx in 0..total_pixels {
            let src_offset = pixel_idx * 4;  // RGBA source
            let dst_offset = pixel_idx * 4;  // ARGB destination
            
            if dst_offset + 3 < size && src_offset + 3 < image_data.len() {
                // Extract RGBA values (branchless for better performance)
                let r = image_data[src_offset] as u32;
                let g = image_data[src_offset + 1] as u32;
                let b = image_data[src_offset + 2] as u32;
                let a = image_data[src_offset + 3] as u32;
                
                // Optimized premultiplication (avoid division where possible)
                let premul_r = ((r * a + 128) >> 8) as u8;  // Fast approximation of r * a / 255
                let premul_g = ((g * a + 128) >> 8) as u8;
                let premul_b = ((b * a + 128) >> 8) as u8;
                
                // Write in ARGB format (little-endian: BGRA in memory)
                unsafe {
                    *buffer.get_unchecked_mut(dst_offset) = premul_b;     // Blue
                    *buffer.get_unchecked_mut(dst_offset + 1) = premul_g; // Green
                    *buffer.get_unchecked_mut(dst_offset + 2) = premul_r; // Red
                    *buffer.get_unchecked_mut(dst_offset + 3) = a as u8;  // Alpha
                }
            }
        }
        
        Ok(())
    }
    
    /// Legacy method kept for fallback compatibility  
    fn write_pixels_to_shm(&self, image: &ImageBuffer<Rgba<u8>, Vec<u8>>, fd: i32, size: usize) -> Result<()> {
        use std::ptr;
        
        // Memory map the file descriptor
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        
        if ptr == libc::MAP_FAILED {
            anyhow::bail!("Failed to mmap SHM file");
        }
        
        // Use optimized writing method
        let result = self.write_pixels_to_persistent_memory(image, ptr as *mut u8, size);
        
        // Unmap memory
        unsafe {
            libc::munmap(ptr, size);
        }
        
        result
    }

    pub fn set_anchor_position(&mut self, position: AnchorPosition, _qh: &QueueHandle<Self>) {
        if let Some(layer_surface) = &self.layer_surface {
            self.anchor_position = position;
            self.last_position_change = Instant::now();
            layer_surface.set_anchor(position.into());
            layer_surface.wl_surface().commit();
            
            info!("Changed anchor position to: {:?}", position);
        }
    }

    /// Check if mouse is near the overlay area
    fn is_mouse_near_overlay(&mut self) -> bool {
        if let Some((x, y, width, height)) = self.get_surface_bounds() {
            //self.mouse_tracker.is_mouse_near_area(x, y, width, height, self.mouse_detection_margin)
            false
        } else {
            false
        }
    }

    fn is_mouse_near_overlay_ap(&mut self, ap: AnchorPosition) -> bool {
        if let Some((x, y, width, height)) = self.get_surface_bounds_ap(ap) {
            //self.mouse_tracker.is_mouse_near_area(x, y, width, height, self.mouse_detection_margin)
            false
        } else {
            false
        }
    }

    /// Check if it's time to automatically reposition the overlay
    pub fn should_auto_reposition(&mut self) -> bool {
        // Immediate repositioning if mouse is detected near overlay
        if self.is_mouse_near_overlay() {
            debug!("Mouse detected near overlay - immediate repositioning triggered");
            return true;
        }
        
        // Fallback: periodic repositioning every interval
        self.last_position_change.elapsed() >= self.auto_reposition_interval
    }

    /// Automatically move to the opposite corner to get out of the way
    pub fn auto_reposition(&mut self, qh: &QueueHandle<Self>) {
        if self.should_auto_reposition() {
            let is_near_overlay = self.is_mouse_near_overlay();
            info!("is_near_overlay = {} (mouse pos = {:?})", is_near_overlay, self.mouse_last_known_pos.read().expect("Cold not read mouse_last_known_pos") );
            let new_position = if is_near_overlay {
                match self.anchor_position { // Flip to opposite side
                    AnchorPosition::LowerLeft => AnchorPosition::LowerRight,
                    AnchorPosition::LowerRight => AnchorPosition::LowerLeft,
                }
            }
            // Is mouse near left/right? Move to opposite.
            else if self.anchor_position == AnchorPosition::LowerLeft && self.ideal_anchor_position == AnchorPosition::LowerRight && !self.is_mouse_near_overlay_ap(AnchorPosition::LowerRight) {
                self.ideal_anchor_position.clone()
            }
            else if self.anchor_position == AnchorPosition::LowerRight && self.ideal_anchor_position == AnchorPosition::LowerLeft && !self.is_mouse_near_overlay_ap(AnchorPosition::LowerLeft) {
                self.ideal_anchor_position.clone()
            }
            else {
                // Keep same position b/c the mouse overlaps the ideal (or we are already there)
                self.anchor_position.clone()
            };

            let reason = if is_near_overlay {
                "mouse detected near overlay"
            } else {
                "periodic auto-repositioning"
            };

            info!("Moving overlay from {:?} to {:?} ({})", 
                  self.anchor_position, new_position, reason);
            self.set_anchor_position(new_position, qh);
        }
    }

    pub fn get_anchor_position(&self) -> AnchorPosition {
        self.anchor_position
    }

    pub fn get_surface_bounds(&self) -> Option<(i32, i32, i32, i32)> {
        if self.layer_surface.is_some() {
            let (x, y) = match self.anchor_position {
                AnchorPosition::LowerLeft => (0, self.get_screen_height() - self.height as i32),
                AnchorPosition::LowerRight => (self.get_screen_width() - self.width as i32, self.get_screen_height() - self.height as i32),
            };
            Some((x, y, self.width as i32, self.height as i32))
        } else {
            None
        }
    }

    pub fn get_surface_bounds_ap(&self, ap: AnchorPosition) -> Option<(i32, i32, i32, i32)> {
        if self.layer_surface.is_some() {
            let (x, y) = match ap {
                AnchorPosition::LowerLeft => (0, self.get_screen_height() - self.height as i32),
                AnchorPosition::LowerRight => (self.get_screen_width() - self.width as i32, self.get_screen_height() - self.height as i32),
            };
            Some((x, y, self.width as i32, self.height as i32))
        } else {
            None
        }
    }


    fn get_screen_width(&self) -> i32 {
        self.outputs.values().map(|o| o.width).max().unwrap_or(1920)
    }

    fn get_screen_height(&self) -> i32 {
        self.outputs.values().map(|o| o.height).max().unwrap_or(1080)
    }
    
    /// Get current file descriptor usage count for monitoring
    pub fn check_fd_usage(&self) -> Result<usize> {
        let proc_fd_dir = format!("/proc/{}/fd", std::process::id());
        match std::fs::read_dir(&proc_fd_dir) {
            Ok(entries) => Ok(entries.count()),
            Err(_) => {
                // Fallback: just report if we have FDs tracked
                Ok(if self.current_shm_fd.is_some() { 1 } else { 0 })
            }
        }
    }
    
    /// Get SHM performance statistics
    pub fn get_shm_stats(&self) -> ShmStats {
        ShmStats {
            persistent_mapping_active: self.persistent_memory_map.is_some(),
            current_buffer_size: self.current_buffer_size,
            max_buffer_size: self.max_buffer_size,
            buffer_utilization: if self.max_buffer_size > 0 {
                (self.current_buffer_size as f32 / self.max_buffer_size as f32 * 100.0) as u8
            } else {
                0
            },
            has_file_descriptor: self.current_shm_fd.is_some(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShmStats {
    pub persistent_mapping_active: bool,
    pub current_buffer_size: u32,
    pub max_buffer_size: u32,
    pub buffer_utilization: u8, // Percentage of max buffer being used
    pub has_file_descriptor: bool,
}

impl Drop for WaylandOverlay {
    fn drop(&mut self) {
        info!("Cleaning up WaylandOverlay resources");
        self.cleanup_shm_resources();
        
        // Additional cleanup for Wayland resources if needed
        if let Some(_layer_surface) = self.layer_surface.take() {
            // Layer surface will be cleaned up by Wayland
        }
        
        debug!("WaylandOverlay cleanup completed");
    }
}

impl CompositorHandler for WaylandOverlay {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for WaylandOverlay {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        let info = OutputInfo {
            width: 1920,
            height: 1080,
        };
        self.outputs.insert(output, info);
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if let Some(output_info) = self.output_state.info(&output) {
            let info = OutputInfo {
                width: output_info.logical_size.map(|s| s.0).unwrap_or(1920),
                height: output_info.logical_size.map(|s| s.1).unwrap_or(1080),
            };
            self.outputs.insert(output, info);
        }
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        self.outputs.remove(&output);
    }
}

impl LayerShellHandler for WaylandOverlay {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        info!("Layer surface closed");
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _layer: &LayerSurface,
        configure: smithay_client_toolkit::shell::wlr_layer::LayerSurfaceConfigure,
        _serial: u32,
    ) {
        self.width = configure.new_size.0;
        self.height = configure.new_size.1;
        
        debug!("Layer surface configured: {}x{}", self.width, self.height);
    }
}


impl ProvidesRegistryState for WaylandOverlay {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    
    registry_handlers![OutputState];
}


// Manual SHM protocol implementation
impl wayland_client::Dispatch<wayland_client::protocol::wl_shm::WlShm, ()> for WaylandOverlay {
    fn event(
        _state: &mut Self,
        _proxy: &wayland_client::protocol::wl_shm::WlShm,
        _event: <wayland_client::protocol::wl_shm::WlShm as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &wayland_client::Connection,
        _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        // Handle SHM events (format announcements, etc.)
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_shm_pool::WlShmPool, ()> for WaylandOverlay {
    fn event(
        _state: &mut Self,
        _proxy: &wayland_client::protocol::wl_shm_pool::WlShmPool,
        _event: <wayland_client::protocol::wl_shm_pool::WlShmPool as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &wayland_client::Connection,
        _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        // Handle SHM pool events
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_buffer::WlBuffer, ()> for WaylandOverlay {
    fn event(
        _state: &mut Self,
        _proxy: &wayland_client::protocol::wl_buffer::WlBuffer,
        _event: <wayland_client::protocol::wl_buffer::WlBuffer as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &wayland_client::Connection,
        _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        // Handle buffer events (release, etc.)
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_region::WlRegion, ()> for WaylandOverlay {
    fn event(
        _state: &mut Self,
        _proxy: &wayland_client::protocol::wl_region::WlRegion,
        _event: <wayland_client::protocol::wl_region::WlRegion as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &wayland_client::Connection,
        _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        // Handle region events
    }
}



delegate_compositor!(WaylandOverlay);
delegate_output!(WaylandOverlay);
delegate_layer!(WaylandOverlay);
delegate_registry!(WaylandOverlay);
