#![cfg(target_arch = "wasm32")]

use crate::errors::{LlmError, Result};

#[link(wasm_import_module = "env")]
extern "C" {
    pub fn host_kv_get(ns_ptr: *const u8, ns_len: u32, key_ptr: *const u8, key_len: u32) -> u64;
    pub fn host_kv_put(
        ns_ptr: *const u8,
        ns_len: u32,
        key_ptr: *const u8,
        key_len: u32,
        val_ptr: *const u8,
        val_len: u32,
    ) -> u32;
    pub fn host_kv_query(ns_ptr: *const u8, ns_len: u32, query_ptr: *const u8, query_len: u32) -> u64;
    pub fn host_vec_search(query_ptr: *const u8, query_len: u32, k: u32) -> u64;
    pub fn host_log(level: u32, msg_ptr: *const u8, msg_len: u32) -> u32;
    pub fn host_now_ms() -> i64;
}

#[inline]
fn unpack(packed: u64) -> (u32, u32) {
    let ptr = (packed & 0xFFFF_FFFF) as u32;
    let len = (packed >> 32) as u32;
    (ptr, len)
}

unsafe fn take_bytes(packed: u64) -> Vec<u8> {
    let (ptr, len) = unpack(packed);
    if ptr == 0 || len == 0 {
        return Vec::new();
    }
    let slice = core::slice::from_raw_parts(ptr as *const u8, len as usize);
    slice.to_vec()
}

pub fn log(msg: &str) {
    let _ = unsafe { host_log(1, msg.as_ptr(), msg.len() as u32) };
}

pub fn now_ms() -> i64 {
    unsafe { host_now_ms() }
}

pub fn kv_get(namespace: &str, key: &str) -> Result<Vec<u8>> {
    let packed = unsafe {
        host_kv_get(
            namespace.as_ptr(),
            namespace.len() as u32,
            key.as_ptr(),
            key.len() as u32,
        )
    };
    Ok(unsafe { take_bytes(packed) })
}

pub fn kv_put(namespace: &str, key: &str, val: &[u8]) -> Result<()> {
    let rc = unsafe {
        host_kv_put(
            namespace.as_ptr(),
            namespace.len() as u32,
            key.as_ptr(),
            key.len() as u32,
            val.as_ptr(),
            val.len() as u32,
        )
    };
    if rc != 0 {
        Ok(())
    } else {
        Err(LlmError::Process(format!("host_kv_put failed rc={}", rc)))
    }
}

pub fn kv_query(namespace: &str, query: &str) -> Result<Vec<u8>> {
    let packed = unsafe {
        host_kv_query(
            namespace.as_ptr(),
            namespace.len() as u32,
            query.as_ptr(),
            query.len() as u32,
        )
    };
    Ok(unsafe { take_bytes(packed) })
}

pub fn vec_search(query: &str, k: u32) -> Result<Vec<u8>> {
    let packed = unsafe { host_vec_search(query.as_ptr(), query.len() as u32, k) };
    Ok(unsafe { take_bytes(packed) })
}
