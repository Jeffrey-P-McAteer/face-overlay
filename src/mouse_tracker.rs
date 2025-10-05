use anyhow::Result;
use std::time::{Duration, Instant};
use tracing::debug;

pub struct MouseTracker {
    last_overlap_time: Option<Instant>,
    flip_delay: Duration,
    is_overlapping: bool,
    should_flip: bool,
}

impl MouseTracker {
    pub fn new(flip_delay_ms: u64) -> Self {
        Self {
            last_overlap_time: None,
            flip_delay: Duration::from_millis(flip_delay_ms),
            is_overlapping: false,
            should_flip: false,
        }
    }

    pub fn update_mouse_position(&mut self, mouse_x: i32, mouse_y: i32, overlay_bounds: (i32, i32, i32, i32)) -> bool {
        let (overlay_x, overlay_y, overlay_width, overlay_height) = overlay_bounds;
        
        let currently_overlapping = mouse_x >= overlay_x
            && mouse_x < overlay_x + overlay_width
            && mouse_y >= overlay_y
            && mouse_y < overlay_y + overlay_height;

        tracing::info!("update_mouse_position {},{} {:?} currently_overlapping = {:?} last_overlap_time = {:?}", mouse_x, mouse_y, overlay_bounds, currently_overlapping, self.last_overlap_time );

        if currently_overlapping && !self.is_overlapping {
            self.last_overlap_time = Some(Instant::now());
            self.is_overlapping = true;
            debug!("Mouse started overlapping overlay at ({}, {})", mouse_x, mouse_y);
        } else if !currently_overlapping && self.is_overlapping {
            self.last_overlap_time = None;
            self.is_overlapping = false;
            self.should_flip = false;
            debug!("Mouse stopped overlapping overlay");
        }

        if self.is_overlapping {
            if let Some(overlap_start) = self.last_overlap_time {
                if overlap_start.elapsed() >= self.flip_delay && !self.should_flip {
                    self.should_flip = true;
                    debug!("🔄 Mouse dwell detected: {}ms >= {}ms, triggering auto-flip", 
                           overlap_start.elapsed().as_millis(), 
                           self.flip_delay.as_millis());
                    return true;
                }
            }
        }

        false
    }


    pub fn reset_flip_state(&mut self) {
        self.should_flip = false;
    }
}


pub struct MouseEventHandler {
    tracker: MouseTracker,
    enabled: bool,
}

impl MouseEventHandler {
    pub fn new(flip_delay_ms: u64, enabled: bool) -> Result<Self> {
        Ok(Self {
            tracker: MouseTracker::new(flip_delay_ms),
            enabled,
        })
    }

    pub fn check_for_flip(&mut self, overlay_bounds: Option<(i32, i32, i32, i32)>, mouse_position: (i32, i32)) -> bool {
        if !self.enabled {
            return false;
        }

        let Some(bounds) = overlay_bounds else {
            return false;
        };

        // Use mouse position from Wayland seat protocol via overlay
        self.tracker.update_mouse_position(mouse_position.0, mouse_position.1, bounds)
    }

    pub fn reset_flip_state(&mut self) {
        self.tracker.reset_flip_state();
    }

}
