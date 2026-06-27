//! Projector output windows and the project-and-capture helper.

use anyhow::Result;
use opencv::{
    core::{Mat, Size},
    highgui::*,
};

use crate::external_camera::ExternalCamera;

/// Window title for projector `id` (one OpenCV window per projector).
pub fn get_winname(id: usize) -> String {
    format!("Pattern {id}")
}

/// Width of the primary display, measured at runtime, in the OS window
/// coordinate space.
///
/// Used as the horizontal offset to place projector windows to the right of the
/// primary monitor (see [`setup_window`]). The value is intentionally *not*
/// multiplied by the display scale factor: OpenCV's `move_window` positions
/// windows in logical coordinates (points), which is the same space
/// `display-info` reports here. Falls back to a sensible default if the displays
/// cannot be queried.
fn primary_display_width() -> i32 {
    const FALLBACK_WIDTH: i32 = 1920;
    let Ok(displays) = display_info::DisplayInfo::all() else {
        return FALLBACK_WIDTH;
    };
    displays
        .iter()
        .find(|d| d.is_primary)
        .or_else(|| displays.first())
        .map(|d| d.width as i32)
        .unwrap_or(FALLBACK_WIDTH)
}

/// Create a full-screen window for projector `id`, placed to the right of the
/// primary display.
pub fn setup_window(id: usize, projector_size: Size) -> Result<()> {
    let winname = get_winname(id);
    named_window(&winname, WINDOW_NORMAL)?;
    move_window(
        &winname,
        primary_display_width() + id as i32 * projector_size.width,
        0,
    )?;
    set_window_property(&winname, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN.into())?;
    Ok(())
}

/// Project `image` on window `win_id`, then capture a frame from `camera`.
pub fn project_and_capture(camera: &dyn ExternalCamera, win_id: usize, image: &Mat) -> Result<Mat> {
    imshow(&get_winname(win_id), &image)?;
    wait_key(800)?;
    let frame = camera.capture()?;
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_primary_display_width() {
        assert!(primary_display_width() > 0);
    }
}
