use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, OnceLock, RwLock};
use std::{io, mem, process, ptr, slice};

use libc::{
    c_int, sigaction, sigemptyset, siginfo_t, sigset_t, ucontext_t, MAP_ANONYMOUS, MAP_FAILED,
    MAP_PRIVATE, PROT_READ, PROT_WRITE, SA_SIGINFO, _SC_PAGESIZE,
};
use rangemap::RangeMap;

fn system_page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    *PAGE_SIZE.get_or_init(|| unsafe { libc::sysconf(_SC_PAGESIZE) as usize })
}

struct Mmap<T> {
    ptr: *mut T,
    len: usize,
    n_bytes: usize,
}

impl<T> Drop for Mmap<T> {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as _, self.n_bytes);
        }
    }
}

impl<T> Mmap<T> {
    fn new(len: usize, read_only: bool) -> io::Result<Self> {
        unsafe {
            let mut n_bytes = len * mem::size_of::<T>();
            let page_size = system_page_size();

            // pad n_bytes to be a multiple of page size
            let n_bytes_rem = n_bytes % page_size;
            if n_bytes_rem != 0 {
                n_bytes += page_size - n_bytes_rem;
            }

            let prot = if read_only {
                PROT_READ
            } else {
                PROT_READ | PROT_WRITE
            };

            let ptr = libc::mmap(
                ptr::null_mut(),
                n_bytes,
                prot,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            );
            if ptr == MAP_FAILED {
                return Err(io::Error::last_os_error());
            }

            Ok(Mmap {
                ptr: ptr as _,
                len,
                n_bytes,
            })
        }
    }
}

impl<T> AsRef<[T]> for Mmap<T> {
    fn as_ref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> AsMut<[T]> for Mmap<T> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

struct MemoryInner {
    bytes: Mmap<u8>,
    dirty_page_bits: Mmap<AtomicBool>,
}

impl MemoryInner {
    fn new(len: usize) -> io::Result<Self> {
        Ok(MemoryInner {
            bytes: Mmap::new(len, true)?,
            dirty_page_bits: Mmap::new(len / system_page_size(), false)?,
        })
    }

    unsafe fn process_segv(&mut self, si_addr: usize) -> io::Result<()> {
        let start_addr = self.bytes.ptr as usize;

        let page_size = system_page_size();
        let page_index = (si_addr - start_addr) / page_size;
        let page_addr = start_addr + page_index * page_size;

        let was_dirty = self.dirty_page_bits.as_mut()[page_index].fetch_or(true, Ordering::SeqCst);

        if !was_dirty && libc::mprotect(page_addr as _, page_size, PROT_READ | PROT_WRITE) != 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(())
    }
}

/// The type of the global memory map.
type MemoryMap = RangeMap<usize, usize>;

/// Global map from a memory address range to a pointer to its `MemoryInner`
/// struct.
fn global_memory_map() -> &'static RwLock<MemoryMap> {
    static MEMORY_MAP: OnceLock<RwLock<MemoryMap>> = OnceLock::new();
    MEMORY_MAP.get_or_init(|| RwLock::new(MemoryMap::new()))
}

fn with_map<T, F>(f: F) -> T
where
    F: FnOnce(&MemoryMap) -> T,
{
    f(&global_memory_map().read().unwrap())
}

fn with_map_mut<T, F>(f: F) -> T
where
    F: FnOnce(&mut MemoryMap) -> T,
{
    f(&mut global_memory_map().write().unwrap())
}

pub struct Memory(&'static mut MemoryInner);

impl Drop for Memory {
    fn drop(&mut self) {
        with_map_mut(|map| {
            let inner_slice = self.0.bytes.as_ref();

            let inner_begin = inner_slice.as_ptr() as usize;
            let inner_end = inner_begin + inner_slice.len();
            let inner_range = inner_begin..inner_end;

            map.remove(inner_range);
            unsafe {
                let _ = Box::from_raw(self.0);
            }
        });
    }
}

impl Memory {
    pub fn new(len: usize) -> io::Result<Self> {
        with_map_mut(|map| {
            setup_action();

            let inner = MemoryInner::new(len)?;

            let inner_slice = inner.bytes.as_ref();

            let inner_begin = inner_slice.as_ptr() as usize;
            let inner_end = inner_begin + inner_slice.len();
            let inner_range = inner_begin..inner_end;

            let leaked_inner = Box::leak(Box::new(inner));
            map.insert(inner_range, leaked_inner as *const MemoryInner as _);

            Ok(Memory(leaked_inner))
        })
    }
}

impl AsRef<[u8]> for Memory {
    fn as_ref(&self) -> &[u8] {
        let inner = &*self.0;
        inner.bytes.as_ref()
    }
}

impl AsMut<[u8]> for Memory {
    fn as_mut(&mut self) -> &mut [u8] {
        let inner = &mut *self.0;
        inner.bytes.as_mut()
    }
}

impl Deref for Memory {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl DerefMut for Memory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

unsafe fn segfault_handler(sig: c_int, info: *mut siginfo_t, ctx: *mut ucontext_t) {
    with_map(move |global_map| {
        let si_addr = (*info).si_addr() as usize;

        if let Some(inner_ptr) = global_map.get(&si_addr) {
            let inner = &mut *(*inner_ptr as *mut MemoryInner);

            if inner.process_segv(si_addr).is_err() {
                call_old_action(sig, info, ctx);
            }

            return;
        }

        call_old_action(sig, info, ctx);
    });
}

unsafe fn call_old_action(sig: c_int, info: *mut siginfo_t, ctx: *mut ucontext_t) {
    let old_act = setup_action();

    // If SA_SIGINFO is set, the old action is a `fn(c_int, *mut siginfo_t, *mut
    // ucontext_t)`. Otherwise, it's a `fn(c_int)`.
    if old_act.sa_flags & SA_SIGINFO == 0 {
        let act: fn(c_int) = mem::transmute(old_act.sa_sigaction);
        act(sig);
    } else {
        let act: fn(c_int, *mut siginfo_t, *mut ucontext_t) = mem::transmute(old_act.sa_sigaction);
        act(sig, info, ctx);
    }
}

static SIGNAL_HANDLER: Once = Once::new();

// Sets up [`segfault_handler`] to handle SIGSEGV, and returns the previous
// action used to handle it, if any.
fn setup_action() -> sigaction {
    static OLD_ACTION: OnceLock<sigaction> = OnceLock::new();

    SIGNAL_HANDLER.call_once(|| {
        unsafe {
            let mut sa_mask = MaybeUninit::<sigset_t>::uninit();
            sigemptyset(sa_mask.as_mut_ptr());

            let act = sigaction {
                sa_sigaction: segfault_handler as _,
                sa_mask: sa_mask.assume_init(),
                sa_flags: SA_SIGINFO,
                #[cfg(target_os = "linux")]
                sa_restorer: None,
            };
            let mut old_act = MaybeUninit::<sigaction>::uninit();

            if libc::sigaction(libc::SIGSEGV, &act, old_act.as_mut_ptr()) != 0 {
                process::exit(1);
            }

            // On Apple Silicon for some reason SIGBUS is thrown instead of SIGSEGV.
            // TODO should investigate properly
            #[cfg(target_os = "macos")]
            if libc::sigaction(libc::SIGBUS, &act, old_act.as_mut_ptr()) != 0 {
                process::exit(2);
            }

            OLD_ACTION.get_or_init(move || old_act.assume_init());
        }
    });

    *OLD_ACTION.get().unwrap()
}

#[test]
fn read_write() {
    let mut memory = Memory::new(4096).unwrap();
    memory[0] = 42;
    assert_eq!(memory[0], 42);
    assert!(memory.0.dirty_page_bits.as_ref()[0].load(Ordering::SeqCst));
}
