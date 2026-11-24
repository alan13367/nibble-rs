//! The default thread count: the performance cores on Apple Silicon, else one
//! thread per physical core.
//!
//! Decoding is limited by memory bandwidth, which the threads of one core
//! share, so a second thread per core (SMT) adds no speed. It costs some:
//! with a thread on every logical CPU, any other process preempts one of
//! them, and an op cannot finish until a preempted task resumes. On a
//! Ryzen 5 5600X, 12 threads decode about a third slower than 6. llama.cpp
//! defaults to physical cores for the same reason. Apple Silicon is the
//! exception: see the macOS `physical_cores`.

/// Threads to use when the caller does not say: the performance cores on
/// Apple Silicon, else the physical cores (where the OS reports them), or else
/// the logical CPUs available to this process, whichever is smaller.
pub(crate) fn default_threads() -> usize {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    physical_cores().map_or(available, |cores| cores.clamp(1, available))
}

/// The number of physical cores, from `GetLogicalProcessorInformationEx`.
#[cfg(windows)]
fn physical_cores() -> Option<usize> {
    /// `RelationProcessorCore`: one record per physical core.
    const RELATION_PROCESSOR_CORE: i32 = 0;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: i32,
            buffer: *mut u8,
            length: *mut u32,
        ) -> i32;
    }

    let mut len = 0u32;
    // SAFETY: with a null buffer the call only writes the required length
    // to `len` (and fails, as expected).
    unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut len)
    };
    if len == 0 {
        return None;
    }
    // u64 elements give the records their natural 8-byte alignment.
    let mut buf = vec![0u64; (len as usize).div_ceil(8)];
    // SAFETY: `buf` has room for `len` bytes, as `len` tells the call.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buf.as_mut_ptr().cast(), &mut len)
    };
    if ok == 0 {
        return None;
    }
    let len = (len as usize).min(8 * buf.len());
    // SAFETY: the first `len` bytes of `buf` are initialized (they are
    // `u64`s), and `u8` has no alignment requirement.
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), len) };
    count_core_records(bytes)
}

/// Counts the `RelationProcessorCore` records in a
/// `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` list: each record starts with
/// its relationship and its size in bytes, both 32-bit.
#[cfg(any(windows, test))]
fn count_core_records(mut bytes: &[u8]) -> Option<usize> {
    let mut cores = 0;
    while let Some((header, _)) = bytes.split_first_chunk::<8>() {
        let relationship = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        if relationship == 0 {
            cores += 1;
        }
        if size < 8 || size > bytes.len() {
            break;
        }
        bytes = &bytes[size..];
    }
    (cores > 0).then_some(cores)
}

/// The number of physical cores: the distinct sets of hardware threads that
/// share a core, from sysfs.
#[cfg(target_os = "linux")]
fn physical_cores() -> Option<usize> {
    let mut cores = std::collections::HashSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = entry.file_name();
        let is_cpu = name
            .to_str()
            .and_then(|n| n.strip_prefix("cpu"))
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
        if !is_cpu {
            continue;
        }
        // Offline CPUs have no topology; they are not ours to use anyway.
        if let Ok(siblings) =
            std::fs::read_to_string(entry.path().join("topology/thread_siblings_list"))
        {
            cores.insert(siblings.trim().to_owned());
        }
    }
    (!cores.is_empty()).then_some(cores.len())
}

/// Performance cores on Apple Silicon, from `hw.perflevel0.physicalcpu`.
///
/// Decoding waits for the slowest task. A task that lands on an efficiency
/// core holds the rest of the op, and those cores share a much smaller cache.
/// On an M4 Max, 10 performance cores decode faster than all 14. llama.cpp
/// defaults to the performance cores for the same reason. Machines without
/// that sysctl (Intel Macs) use `hw.physicalcpu` instead.
#[cfg(target_os = "macos")]
fn physical_cores() -> Option<usize> {
    sysctl_usize(c"hw.perflevel0.physicalcpu").or_else(|| sysctl_usize(c"hw.physicalcpu"))
}

/// `sysctlbyname` of an integer, or `None` when the name is missing or zero.
#[cfg(target_os = "macos")]
fn sysctl_usize(name: &std::ffi::CStr) -> Option<usize> {
    let mut buf = [0u8; 8];
    let mut len = buf.len();
    // SAFETY: `name` is a NUL-terminated C string, `buf` has room for `len`
    // bytes, and a failed call is reported by a non-zero return. Only the
    // `len` bytes the call says it wrote are read afterwards.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len == 0 || len > buf.len() {
        return None;
    }
    let value = match len {
        4 => u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize,
        8 => usize::from_le_bytes(buf),
        _ => return None,
    };
    (value > 0).then_some(value)
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    /// Reads a sysctl by name. `oldlenp` is in/out: the buffer size on entry,
    /// the written size on success.
    ///
    /// # Safety
    ///
    /// `name` must be a valid NUL-terminated string. `oldp` must be either
    /// null or valid for writes of `*oldlenp` bytes. `oldlenp` must be a
    /// valid pointer. `newp` / `newlen` describe an optional new value; both
    /// are null and zero when only reading.
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> i32;
}

/// Elsewhere the logical CPU count stands in for the cores.
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn physical_cores() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_between_one_and_the_logical_cpu_count() {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        let threads = default_threads();
        assert!(
            (1..=available).contains(&threads),
            "{threads} of {available}"
        );
    }

    #[test]
    fn core_records_are_counted_by_relationship_and_size() {
        let record = |relationship: u32, size: u32| {
            let mut r = relationship.to_le_bytes().to_vec();
            r.extend(size.to_le_bytes());
            r.resize(size as usize, 0);
            r
        };
        let list: Vec<u8> = [record(0, 48), record(2, 80), record(0, 48), record(0, 48)].concat();
        assert_eq!(count_core_records(&list), Some(3));
        assert_eq!(count_core_records(&[]), None);
        // A corrupt size stops the walk instead of looping or overrunning.
        assert_eq!(count_core_records(&record(0, 4)[..4]), None);
        let mut bad = record(0, 48);
        bad[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(count_core_records(&bad), Some(1));
    }
}
