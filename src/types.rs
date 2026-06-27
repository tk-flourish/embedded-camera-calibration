//! Shared data types for the calibration pipeline: identifiers, per-camera
//! decoded correspondences, calibration results, and serde adapters for OpenCV
//! point types (which are not `Serialize`/`Deserialize` themselves).

use std::collections::BTreeMap;

use opencv::core::{Point2f, Point2i};
use serde::{Deserialize, Serialize};

/// Parse an `x,y` pair of millimetres into a [`Point2f`].
///
/// Used as a clap `value_parser` for rig-specific board coordinates supplied on
/// the command line (e.g. the embedded-camera positions).
pub fn parse_board_point_mm(s: &str) -> Result<Point2f, String> {
    let (x, y) = s
        .split_once(',')
        .ok_or_else(|| format!("expected `x,y` (mm), got `{s}`"))?;
    let x = x
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("invalid x in `{s}`: {e}"))?;
    let y = y
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("invalid y in `{s}`: {e}"))?;
    Ok(Point2f::new(x, y))
}

/// Identifies which projector (`0..projector_count`); this is the projector
/// identity encoded in the Gray-code ID bits and decoded by each camera.
pub type ProjectorId = u32;
/// Identifies an embedded camera within the rig (its index in the camera list).
pub type CameraId = usize;

/// A single decoded correspondence reported by an embedded camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CameraResult {
    /// Projector-image pixel illuminating this camera's optical center
    /// (the paper's p_m(n)).
    pub camera_position_in_projector: Point2i,
    /// Camera-image pixel (centroid) that received projector m's light
    /// (the paper's c_n(m)).
    pub projector_position_in_camera: Point2i,
}

/// Correspondences for one projector point across all cameras that saw it.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationResultEntry {
    /// Decoded result per camera ID for this projector point.
    pub cameras: BTreeMap<CameraId, CameraResult>,
}

/// Full calibration result: one entry per projector point.
pub type CalibrationResult = Vec<CalibrationResultEntry>;

/// serde adapter for `Point2i`, used via `#[serde(with = ...)]`.
/// Represents the point as an `(x, y)` tuple on the wire.
mod point_i32_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::*;

    /// Serialize a `Point2i` as an `(x, y)` tuple.
    pub fn serialize<S>(p: &Point2i, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (p.x, p.y).serialize(s)
    }

    /// Deserialize a `Point2i` from an `(x, y)` tuple.
    pub fn deserialize<'de, D>(d: D) -> Result<Point2i, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (x, y) = <(i32, i32)>::deserialize(d)?;
        Ok(Point2i { x, y })
    }
}

/// serde adapter for `Point2f`, used via `#[serde(with = ...)]`.
/// Represents the point as an `(x, y)` tuple on the wire.
mod point_f32_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    use super::*;

    /// Serialize a `Point2f` as an `(x, y)` tuple.
    pub fn serialize<S>(p: &Point2f, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (p.x, p.y).serialize(s)
    }

    /// Deserialize a `Point2f` from an `(x, y)` tuple.
    pub fn deserialize<'de, D>(d: D) -> Result<Point2f, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (x, y) = <(f32, f32)>::deserialize(d)?;
        Ok(Point2f { x, y })
    }
}

/// One mesh control point pairing a projector position seen by an embedded
/// camera with that camera's effective position on the board, used to model
/// optical-center compensation. These are the {c_n(k), x_n(k)} pairs that fit
/// the per-camera homography M_n in the paper. Serializable for caching the
/// mesh to disk.
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct CameraMeshPointPair {
    /// Camera-image pixel observing the projector's pixel (the paper's c_n(k)).
    #[serde(with = "point_i32_serde")]
    pub projector_position_in_embedded_camera: Point2i,

    /// Where that ray meets the board, in world coordinates (mm) — x_n(k).
    #[serde(with = "point_f32_serde")]
    pub effective_camera_position_in_world: Point2f,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_plain_pair() {
        let p = parse_board_point_mm("211.32,32.64").unwrap();
        assert_eq!(p.x, 211.32);
        assert_eq!(p.y, 32.64);
    }

    #[test]
    fn tolerates_surrounding_whitespace() {
        let p = parse_board_point_mm(" 1 , 2 ").unwrap();
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }

    #[test]
    fn parses_negative_values() {
        let p = parse_board_point_mm("-1.5,-2.25").unwrap();
        assert_eq!(p.x, -1.5);
        assert_eq!(p.y, -2.25);
    }

    #[test]
    fn rejects_missing_comma() {
        assert!(parse_board_point_mm("12").is_err());
    }

    #[test]
    fn rejects_non_numeric() {
        assert!(parse_board_point_mm("a,b").is_err());
    }

    #[test]
    fn rejects_extra_component() {
        // The second `split_once` half ("2,3") is not a valid f32.
        assert!(parse_board_point_mm("1,2,3").is_err());
    }

    #[test]
    fn rejects_empty() {
        assert!(parse_board_point_mm("").is_err());
    }
}
