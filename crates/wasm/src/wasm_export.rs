#![cfg(target_arch = "wasm32")]

use crate::dispatch::{dispatch_json, LearnSession};
use crate::graph::host_kv_backend::HostKv;
use std::sync::Mutex;

static SESSION: Mutex<Option<LearnSession<HostKv>>> = Mutex::new(None);

#[no_mangle]
pub extern "C" fn rs_learn_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

// SAFETY CONTRACT: rs_learn_free MUST be called with the exact same `ptr`
// and `len` that rs_learn_alloc/rs_learn_dispatch returned/reported for this
// allocation. It reconstructs the Vec via Vec::from_raw_parts(ptr, len, len),
// using `len` as both the reconstructed length and capacity, matching
// rs_learn_alloc's `with_capacity(len)` and leak_bytes' packed length. A
// mismatched len is undefined behavior (wrong capacity handed to the
// deallocator). The host must echo back the original alloc/dispatch length
// verbatim, never a derived "bytes actually used" count.
#[no_mangle]
pub extern "C" fn rs_learn_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 { return; }
    unsafe { let _ = Vec::from_raw_parts(ptr, len, len); }
}

fn pack(ptr: *const u8, len: u32) -> u64 {
    ((ptr as u64) & 0xFFFF_FFFF) | ((len as u64) << 32)
}

fn leak_bytes(bytes: Vec<u8>) -> u64 {
    if bytes.is_empty() { return 0; }
    let len = bytes.len() as u32;
    let mut v = bytes;
    let ptr = v.as_mut_ptr();
    core::mem::forget(v);
    pack(ptr, len)
}

#[no_mangle]
pub extern "C" fn rs_learn_dispatch(ptr: *const u8, len: usize) -> u64 {
    if ptr.is_null() || len == 0 { return 0; }
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    let mut guard = match SESSION.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.is_none() {
        *guard = Some(LearnSession::new(HostKv));
    }
    let session = guard.as_mut().unwrap();
    let resp = dispatch_json(session, bytes);
    leak_bytes(resp)
}

#[no_mangle]
pub extern "C" fn rs_learn_version() -> *const u8 {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr()
}
