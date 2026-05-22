pub mod core;

#[cfg(target_arch = "wasm32")]
pub mod persist;

pub use core::{Router, Route, RouteCtx, RouterConfig, SEED};
