use anyhow::{Context, Result};
use image::{ImageBuffer, Rgba};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
        },
        WaylandSurface,
    },
    shm::{Shm, ShmHandler},
};
use std::collections::HashMap;
use tracing::{debug, info, warn};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_buffer, wl_output, wl_surface},
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
    shm: Shm,
    layer_shell: LayerShell,
    
    layer_surface: Option<LayerSurface>,
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
    pub fn new(anchor_position: AnchorPosition) -> Result<(Self, Connection, wayland_client::EventQueue<Self>)> {
        let conn = Connection::connect_to_env()
            .context("Failed to connect to Wayland compositor")?;

        let (globals, event_queue) = registry_queue_init(&conn)
            .context("Failed to initialize Wayland registry")?;

        let qh = event_queue.handle();

        let registry_state = RegistryState::new(&globals);
        let output_state = OutputState::new(&globals, &qh);
        let compositor_state = CompositorState::bind(&globals, &qh)
            .context("Compositor not available")?;
        
        let shm = Shm::bind(&globals, &qh)
            .context("Shared memory not available")?;
        
        let layer_shell = LayerShell::bind(&globals, &qh)
            .context("Layer shell not available")?;

        info!("Connected to Wayland compositor with layer shell support");

        let overlay = Self {
            registry_state,
            output_state,
            compositor_state,
            shm,
            layer_shell,
            layer_surface: None,
            buffer: None,
            width: 640,
            height: 480,
            anchor_position,
            outputs: HashMap::new(),
        };

        Ok((overlay, conn, event_queue))
    }

    pub fn create_layer_surface(&mut self, qh: &QueueHandle<Self>) -> Result<()> {
        let surface = self.compositor_state.create_surface(qh);
        
        let layer_surface = self.layer_shell.create_layer_surface(
            qh,
            surface,
            Layer::Overlay,
            Some("face-overlay"),
            None,
        );

        layer_surface.set_anchor(self.anchor_position.into());
        layer_surface.set_exclusive_zone(-1);
        layer_surface.set_margin(20, 20, 20, 20);
        layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer_surface.set_size(self.width, self.height);

        let surface_clone = layer_surface.wl_surface().clone();
        surface_clone.commit();
        
        self.layer_surface = Some(layer_surface);
        
        info!("Created layer surface with anchor: {:?}", self.anchor_position);
        
        Ok(())
    }

    pub fn update_frame(&mut self, image: &ImageBuffer<Rgba<u8>, Vec<u8>>, _qh: &QueueHandle<Self>) -> Result<()> {
        let Some(layer_surface) = &self.layer_surface else {
            warn!("No layer surface available for frame update");
            return Ok(());
        };

        let (img_width, img_height) = image.dimensions();
        
        if img_width != self.width || img_height != self.height {
            self.width = img_width;
            self.height = img_height;
            layer_surface.set_size(self.width, self.height);
        }

        // Simplified buffer creation - proper implementation would use actual SHM pools
        warn!("Using placeholder buffer implementation");
        
        debug!("Would update frame: {}x{}", self.width, self.height);

        debug!("Updated frame: {}x{}", self.width, self.height);
        
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

impl ShmHandler for WaylandOverlay {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for WaylandOverlay {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    
    registry_handlers![OutputState];
}

delegate_compositor!(WaylandOverlay);
delegate_output!(WaylandOverlay);
delegate_shm!(WaylandOverlay);
delegate_layer!(WaylandOverlay);
delegate_registry!(WaylandOverlay);