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
use tracing::{debug, info, warn};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_buffer, wl_output, wl_shm, wl_surface},
    Connection, QueueHandle,
};

#[derive(Clone, Copy, Debug)]
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
    
    outputs: HashMap<wl_output::WlOutput, OutputInfo>,
}

#[derive(Debug, Clone)]
struct OutputInfo {
    name: String,
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
            outputs: HashMap::new(),
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
        layer_surface.set_margin(20, 20, 20, 20);
        layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer_surface.set_size(self.width, self.height);

        let surface_clone = layer_surface.wl_surface().clone();
        surface_clone.commit();
        
        self.layer_surface = Some(layer_surface);
        
        info!("Layer surface created successfully with anchor: {:?}", self.anchor_position);
        info!("Surface should appear in {} corner with 20px margins", 
              match self.anchor_position {
                  AnchorPosition::LowerLeft => "lower-left",
                  AnchorPosition::LowerRight => "lower-right",
              });
        
        Ok(())
    }

    pub fn update_frame(&mut self, image: &ImageBuffer<Rgba<u8>, Vec<u8>>, qh: &QueueHandle<Self>) -> Result<()> {
        let Some(layer_surface) = &self.layer_surface else {
            warn!("No layer surface available for frame update");
            return Ok(());
        };

        let (img_width, img_height) = image.dimensions();
        
        // Update surface size if image dimensions changed
        if img_width != self.width || img_height != self.height {
            self.width = img_width;
            self.height = img_height;
            layer_surface.set_size(self.width, self.height);
            
            // Size changed - would need to recreate SHM pool in full implementation
            debug!("Layer surface size changed to {}x{}", self.width, self.height);
        }

        // Create shared memory buffer for actual pixel display
        let stride = self.width * 4; // 4 bytes per pixel (ARGB)
        let buffer_size = stride * self.height;
        
        // Create a simple shared memory buffer using memfd
        let shm_fd = self.create_shm_fd(buffer_size as usize)?;
        
        // Create Wayland SHM pool
        use std::os::fd::BorrowedFd;
        let borrowed_fd = unsafe { BorrowedFd::borrow_raw(shm_fd) };
        let pool = self.shm.create_pool(borrowed_fd, buffer_size as i32, qh, ());
        
        // Create buffer from pool
        let buffer = pool.create_buffer(
            0,                           // offset
            self.width as i32,          // width
            self.height as i32,         // height  
            stride as i32,              // stride
            wl_shm::Format::Argb8888,   // format
            qh,
            (),
        );
        
        // Write pixel data to shared memory
        self.write_pixels_to_shm(image, shm_fd, buffer_size as usize)?;
        
        // Attach buffer and commit
        let surface = layer_surface.wl_surface();
        surface.attach(Some(&buffer), 0, 0);
        surface.damage(0, 0, self.width as i32, self.height as i32);
        surface.commit();
        
        // Clean up old buffer
        if let Some(old_buffer) = self.buffer.take() {
            old_buffer.destroy();
        }
        if let Some(old_pool) = self.pool.take() {
            old_pool.destroy();
        }
        
        // Store new resources
        self.buffer = Some(buffer);
        self.pool = Some(pool);
        
        debug!("Frame updated: {}x{} with actual pixel data displayed", self.width, self.height);
        
        Ok(())
    }

    fn create_shm_fd(&self, size: usize) -> Result<i32> {
        use std::ffi::CString;
        use std::os::unix::io::AsRawFd;
        
        // Create anonymous file descriptor using memfd_create
        let name = CString::new("face-overlay-shm").unwrap();
        let fd = unsafe {
            libc::syscall(libc::SYS_memfd_create, name.as_ptr(), libc::MFD_CLOEXEC)
        };
        
        if fd == -1 {
            // Fallback to tmpfile if memfd_create fails
            let file = tempfile::tempfile()
                .context("Failed to create temporary file for SHM")?;
            let fd = file.as_raw_fd();
            
            // Truncate to required size
            unsafe {
                if libc::ftruncate(fd, size as i64) == -1 {
                    anyhow::bail!("Failed to resize SHM file");
                }
            }
            
            // Leak the file handle to keep it alive
            std::mem::forget(file);
            Ok(fd)
        } else {
            let fd = fd as i32;
            
            // Truncate to required size
            unsafe {
                if libc::ftruncate(fd, size as i64) == -1 {
                    anyhow::bail!("Failed to resize SHM file");
                }
            }
            
            Ok(fd)
        }
    }
    
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
        
        // Write pixel data
        let (img_width, img_height) = image.dimensions();
        let buffer = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size) };
        
        for y in 0..img_height {
            for x in 0..img_width {
                let pixel = image.get_pixel(x, y);
                let offset = ((y * img_width + x) * 4) as usize;
                
                if offset + 3 < size {
                    // Convert RGBA to ARGB format (Wayland's preferred format)
                    // With premultiplied alpha for correct blending
                    let r = pixel[0] as u32;
                    let g = pixel[1] as u32;
                    let b = pixel[2] as u32;
                    let a = pixel[3] as u32;
                    
                    // Premultiply RGB values by alpha
                    let premul_r = if a > 0 { (r * a / 255) as u8 } else { 0 };
                    let premul_g = if a > 0 { (g * a / 255) as u8 } else { 0 };
                    let premul_b = if a > 0 { (b * a / 255) as u8 } else { 0 };
                    
                    // Write in ARGB format (little-endian: BGRA in memory)
                    buffer[offset] = premul_b;     // Blue
                    buffer[offset + 1] = premul_g; // Green
                    buffer[offset + 2] = premul_r; // Red
                    buffer[offset + 3] = a as u8;  // Alpha
                }
            }
        }
        
        // Unmap memory
        unsafe {
            libc::munmap(ptr, size);
        }
        
        Ok(())
    }

    pub fn set_anchor_position(&mut self, position: AnchorPosition, _qh: &QueueHandle<Self>) {
        if let Some(layer_surface) = &self.layer_surface {
            self.anchor_position = position;
            layer_surface.set_anchor(position.into());
            layer_surface.wl_surface().commit();
            
            info!("Changed anchor position to: {:?}", position);
        }
    }

    pub fn get_anchor_position(&self) -> AnchorPosition {
        self.anchor_position
    }

    pub fn get_surface_bounds(&self) -> Option<(i32, i32, i32, i32)> {
        if self.layer_surface.is_some() {
            let (x, y) = match self.anchor_position {
                AnchorPosition::LowerLeft => (20, self.get_screen_height() - self.height as i32 - 20),
                AnchorPosition::LowerRight => (self.get_screen_width() - self.width as i32 - 20, self.get_screen_height() - self.height as i32 - 20),
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
            name: "Unknown".to_string(),
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
                name: output_info.name.clone().unwrap_or_else(|| "Unknown".to_string()),
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

delegate_compositor!(WaylandOverlay);
delegate_output!(WaylandOverlay);
delegate_layer!(WaylandOverlay);
delegate_registry!(WaylandOverlay);