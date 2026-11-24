//! A persistent pool of worker threads for the CPU ops.
//!
//! A forward pass runs about a hundred parallel ops per token, many of them
//! only tens of microseconds long, so the cost of handing work to other
//! threads decides how much threading helps. The workers of a [`ThreadPool`]
//! live as long as the pool (one per session) and take jobs through a few
//! atomics instead of being spawned per op or per token.
//!
//! # Protocol
//!
//! - **Publish.** [`ThreadPool::run`] stores a pointer to the job (a task
//!   count and the caller's closure), resets the task counter and increments
//!   `epoch`; job `e` is the one published by the increment to `e`.
//! - **Join.** A worker that sees epoch `e` increments `active`, then checks
//!   that job `e` is not `closed` yet. If it is, the worker leaves without
//!   touching it.
//! - **Work.** Every thread in the job, the caller included, claims task
//!   indices from the shared counter until none are left.
//! - **Close.** When the caller can claim no more tasks, it marks job `e`
//!   closed and waits for `active` to drop to zero, also when a task
//!   panicked. A worker that joined before the close is waited for; one that
//!   joins after it sees the close and leaves. So the borrowed closure
//!   outlives every use of it, which is what makes the lifetime erasure in
//!   [`Job`] sound.
//!
//! The caller never waits for a worker that did not join: a worker that the
//! OS has descheduled (likely when there are as many threads as logical
//! CPUs) delays a job by at most the task it is running, not until it is
//! scheduled again.
//!
//! The join check and the close are a store-then-load pattern on two
//! locations (`active`, then `closed`; `closed`, then `active`), as is the
//! park handshake below. All four accesses of each are `SeqCst`, so at
//! least one side sees the other's store.
//!
//! **Idle.** Between jobs a worker spins on `epoch` for a while, which
//! covers the gaps between the ops of a forward pass, then parks: it sets
//! its `sleeping` flag and re-reads `epoch`, while `run` increments `epoch`
//! and then unparks every worker whose flag is set. Wakeups are per thread
//! ([`std::thread::Thread::unpark`]); a shared condition variable would
//! convoy every wakeup through one lock. A stale unpark only makes one later
//! `park` return early.

use std::any::Any;
use std::hint::spin_loop;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// How long an idle worker spins before parking. It covers the serial gaps
/// inside a forward pass (norms, activation encoding, a small matvec on the
/// calling thread), so workers park between tokens, not between ops.
/// A worker that parks costs the next op a wakeup, tens of microseconds on
/// Windows.
const SPIN: Duration = Duration::from_micros(500);

/// Spins between two reads of the clock while waiting.
const SPINS_PER_CLOCK_READ: u32 = 64;

/// A fixed set of worker threads that run tasks together with the thread
/// that calls [`ThreadPool::run`].
pub(crate) struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

/// A job as the workers see it: a type-erased pointer to the caller's
/// closure and the function that calls it.
struct Job {
    /// The caller's `F`, borrowed for the duration of [`ThreadPool::run`].
    data: *const (),
    /// `call::<F>`.
    call: unsafe fn(*const (), usize),
    n_tasks: usize,
}

/// Calls the `F` behind `data` with `task`.
///
/// # Safety
///
/// `data` must point to a live `F`.
unsafe fn call<F: Fn(usize) + Sync>(data: *const (), task: usize) {
    // SAFETY: guaranteed by the caller.
    let f = unsafe { &*data.cast::<F>() };
    f(task);
}

/// Keeps a frequently written atomic on its own cache line, so spinning
/// readers of one counter are not disturbed by writes to another.
#[repr(align(128))]
struct Padded<T>(T);

impl<T> std::ops::Deref for Padded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

/// State shared by the pool handle and its workers.
struct Shared {
    /// Incremented to publish a job (or the shutdown). Workers wait for it to change.
    epoch: Padded<AtomicUsize>,
    /// The epoch of the last job closed to new workers.
    closed: Padded<AtomicUsize>,
    /// The next unclaimed task of the current job.
    next: Padded<AtomicUsize>,
    /// Workers that have joined a job and not left it yet.
    active: Padded<AtomicUsize>,
    /// The current job; points into `run`'s stack frame while it is active.
    job: AtomicPtr<Job>,
    /// Set while a job is running, so that a nested or concurrent `run` runs
    /// its tasks inline instead of corrupting the current job.
    busy: AtomicBool,
    shutdown: AtomicBool,
    /// One flag per worker: set while the worker is parked or about to park.
    sleeping: Box<[Padded<AtomicBool>]>,
    /// The first panic raised by a task of the current job.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    spin: Duration,
}

impl ThreadPool {
    /// A pool in which `threads` threads run tasks: the caller of
    /// [`ThreadPool::run`] plus `threads - 1` workers (none for 0 or 1).
    ///
    /// If the system cannot start a worker, the pool keeps the ones it has;
    /// on targets without threads it runs everything on the caller.
    pub(crate) fn new(threads: usize) -> Self {
        Self::with_spin(threads, SPIN)
    }

    /// As [`ThreadPool::new`], with idle workers spinning for `spin` before
    /// they park.
    pub(crate) fn with_spin(threads: usize, spin: Duration) -> Self {
        let n_workers = threads.saturating_sub(1);
        let shared = Arc::new(Shared {
            epoch: Padded(AtomicUsize::new(0)),
            closed: Padded(AtomicUsize::new(0)),
            next: Padded(AtomicUsize::new(0)),
            active: Padded(AtomicUsize::new(0)),
            job: AtomicPtr::new(std::ptr::null_mut()),
            busy: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            sleeping: (0..n_workers)
                .map(|_| Padded(AtomicBool::new(false)))
                .collect(),
            panic: Mutex::new(None),
            spin,
        });
        let mut workers = Vec::with_capacity(n_workers);
        for index in 0..n_workers {
            let worker_shared = Arc::clone(&shared);
            let spawned = thread::Builder::new()
                .name(format!("nibble-worker-{}", index + 1))
                .spawn(move || worker_shared.worker(index));
            match spawned {
                Ok(handle) => workers.push(handle),
                // Out of threads (or none on this target): run narrower.
                Err(_) => break,
            }
        }
        Self { shared, workers }
    }

    /// Threads that run tasks: the workers plus the calling thread.
    pub(crate) fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Calls `f(task)` once for every `task` in `0..n_tasks`, spread over
    /// the pool's threads (the calling thread included), and returns when
    /// every call has returned.
    ///
    /// Tasks run inline, in order, when there is at most one task, no
    /// worker, or the pool is already running a job (a nested call from a
    /// task, or a concurrent call from another thread).
    ///
    /// # Panics
    ///
    /// If a task panics, the first panic is resumed on the calling thread
    /// once every other running task has returned; the pool stays usable.
    pub(crate) fn run<F: Fn(usize) + Sync>(&self, n_tasks: usize, f: F) {
        if n_tasks <= 1 || self.workers.is_empty() {
            (0..n_tasks).for_each(f);
            return;
        }
        let s = &*self.shared;
        if s.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            (0..n_tasks).for_each(f);
            return;
        }
        let job = Job {
            data: (&raw const f).cast(),
            call: call::<F>,
            n_tasks,
        };
        // The previous job is closed and no worker is in it (`active` was
        // zero), so no worker reads these until the new epoch below
        // publishes them.
        s.next.store(0, Ordering::Relaxed);
        s.job.store((&raw const job).cast_mut(), Ordering::Relaxed);
        let epoch = s.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        for (handle, sleeping) in self.workers.iter().zip(&s.sleeping) {
            if sleeping.load(Ordering::SeqCst) {
                handle.thread().unpark();
            }
        }
        s.work(&job);
        // Every task is claimed. Close the job and wait for the workers in
        // it to finish their tasks.
        s.closed.store(epoch, Ordering::SeqCst);
        let mut spins = 0u32;
        while s.active.load(Ordering::SeqCst) != 0 {
            // Spinning is right while workers finish their last task; yield
            // in case one of them was descheduled.
            if spins < 1 << 12 {
                spins += 1;
                spin_loop();
            } else {
                thread::yield_now();
            }
        }
        // No worker is in the job or can join it: `job` and `f` are unused.
        let panic = s
            .panic
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        s.busy.store(false, Ordering::Release);
        if let Some(payload) = panic {
            panic::resume_unwind(payload);
        }
    }

    /// Splits each slice of `data` into chunks of `chunk_len` elements (the
    /// last chunk of a slice may be shorter) and calls
    /// `f(slice, chunk, elements)` for every chunk, in parallel. `chunk` is
    /// the chunk's index within its slice, so it covers elements
    /// `chunk * chunk_len..` of `data[slice]`.
    ///
    /// # Panics
    ///
    /// If `chunk_len` is zero, or as [`ThreadPool::run`].
    pub(crate) fn for_each_chunk<T, F, const N: usize>(
        &self,
        data: [&mut [T]; N],
        chunk_len: usize,
        f: F,
    ) where
        T: Send,
        F: Fn(usize, usize, &mut [T]) + Sync,
    {
        assert!(chunk_len > 0, "chunk length must be non-zero");
        // Task t belongs to the slice whose range of task numbers holds it.
        let mut first_task = [0; N];
        let mut n_tasks = 0;
        for (first, slice) in first_task.iter_mut().zip(&data) {
            *first = n_tasks;
            n_tasks += slice.len().div_ceil(chunk_len);
        }
        let data = data.map(|s| SharedSlice::new(s));
        self.run(n_tasks, |task| {
            let slice = first_task.partition_point(|&first| first <= task) - 1;
            let chunk = task - first_task[slice];
            let start = chunk * chunk_len;
            let range = start..(start + chunk_len).min(data[slice].len);
            // SAFETY: each task number maps to exactly one (slice, chunk)
            // pair, and chunks of one slice do not overlap.
            f(slice, chunk, unsafe { data[slice].get_mut(range) });
        });
    }

    /// Calls `f(i, a_i, b_i)` for every `i`, in parallel, where `a_i` is
    /// chunk `i` of `a` (`a_len` elements) and `b_i` chunk `i` of `b`
    /// (`b_len` elements); the last chunk of each may be shorter. Used for
    /// per-head work over two per-head buffers, and for products whose rows
    /// go to two outputs.
    ///
    /// # Panics
    ///
    /// If a chunk length is zero or the slices do not split into the same
    /// number of chunks, or as [`ThreadPool::run`].
    pub(crate) fn for_each_chunk_pair<A, B, F>(
        &self,
        a: &mut [A],
        a_len: usize,
        b: &mut [B],
        b_len: usize,
        f: F,
    ) where
        A: Send,
        B: Send,
        F: Fn(usize, &mut [A], &mut [B]) + Sync,
    {
        assert!(a_len > 0 && b_len > 0, "chunk lengths must be non-zero");
        let n = a.len().div_ceil(a_len);
        assert_eq!(
            n,
            b.len().div_ceil(b_len),
            "slices of {} and {} elements do not split into as many chunks of {a_len} and {b_len}",
            a.len(),
            b.len(),
        );
        let (a, b) = (SharedSlice::new(a), SharedSlice::new(b));
        self.run(n, |i| {
            let a_range = i * a_len..((i + 1) * a_len).min(a.len);
            let b_range = i * b_len..((i + 1) * b_len).min(b.len);
            // SAFETY: task `i` alone takes chunk `i` of each slice, and the
            // chunks of a slice do not overlap.
            let (ai, bi) = unsafe { (a.get_mut(a_range), b.get_mut(b_range)) };
            f(i, ai, bi);
        });
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        let s = &*self.shared;
        s.shutdown.store(true, Ordering::Relaxed);
        s.epoch.fetch_add(1, Ordering::SeqCst);
        for handle in &self.workers {
            handle.thread().unpark();
        }
        for handle in self.workers.drain(..) {
            // Workers catch task panics, so they always exit cleanly.
            let _ = handle.join();
        }
    }
}

impl std::fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadPool")
            .field("threads", &self.threads())
            .finish_non_exhaustive()
    }
}

impl Shared {
    /// A worker's life: wait for a job, join it unless it is already
    /// closed, help run it, leave, repeat.
    fn worker(&self, index: usize) {
        let mut seen = 0;
        loop {
            let epoch = self.wait(seen, index);
            seen = epoch;
            if self.shutdown.load(Ordering::Relaxed) {
                return;
            }
            self.active.fetch_add(1, Ordering::SeqCst);
            if self.closed.load(Ordering::SeqCst) < epoch {
                let job = self.job.load(Ordering::Relaxed);
                // SAFETY: `run` stored `job` before the increment to
                // `epoch` that this worker acquired. The job was not closed
                // after this worker joined it, so `run` waits for the
                // decrement below before it frees the job or reuses `job`.
                self.work(unsafe { &*job });
            }
            self.active.fetch_sub(1, Ordering::Release);
        }
    }

    /// Waits until `epoch` differs from `seen` and returns it: spinning
    /// first, then parked.
    fn wait(&self, seen: usize, index: usize) -> usize {
        let mut spins = 0u32;
        let mut spin_start = None;
        loop {
            let epoch = self.epoch.load(Ordering::Acquire);
            if epoch != seen {
                return epoch;
            }
            spin_loop();
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(SPINS_PER_CLOCK_READ) {
                let start = *spin_start.get_or_insert_with(Instant::now);
                if start.elapsed() >= self.spin {
                    break;
                }
            }
        }
        let sleeping = &self.sleeping[index];
        loop {
            sleeping.store(true, Ordering::SeqCst);
            let epoch = self.epoch.load(Ordering::SeqCst);
            if epoch != seen {
                sleeping.store(false, Ordering::Relaxed);
                return epoch;
            }
            thread::park();
            sleeping.store(false, Ordering::Relaxed);
        }
    }

    /// Claims and runs tasks of `job` until none are left. A panic is
    /// recorded for `run` to resume, so the caller of `work` always gets to
    /// report completion.
    fn work(&self, job: &Job) {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            loop {
                let task = self.next.fetch_add(1, Ordering::Relaxed);
                if task >= job.n_tasks {
                    break;
                }
                // SAFETY: `job.data` points to the closure `job.call` was
                // instantiated for, alive for the duration of the job.
                unsafe { (job.call)(job.data, task) };
            }
        }));
        if let Err(payload) = result {
            // Leave the remaining tasks unclaimed; the job fails anyway.
            self.next.store(job.n_tasks, Ordering::Relaxed);
            self.panic
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(payload);
        }
    }
}

/// A mutable slice that tasks write disjoint parts of.
struct SharedSlice<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: a `SharedSlice` only hands out the parts of the slice that its
// users guarantee are disjoint (see `get_mut`), so sharing it across threads
// is sharing disjoint `&mut [T]`s, which needs `T: Send`.
unsafe impl<T: Send> Sync for SharedSlice<T> {}

impl<T> SharedSlice<T> {
    fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Elements `range` of the slice.
    ///
    /// # Safety
    ///
    /// The slice this was made from must still be mutably borrowed for the
    /// lifetime chosen for the result, and no two live results may overlap.
    ///
    /// # Panics
    ///
    /// If `range` is out of bounds.
    unsafe fn get_mut<'a>(&self, range: Range<usize>) -> &'a mut [T] {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "range {range:?} out of bounds for a slice of {} elements",
            self.len
        );
        // SAFETY: in bounds (checked above) and, by the caller's guarantee,
        // borrowed and not aliased.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(range.start), range.len()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `n_tasks` tasks and returns how often each one ran.
    fn task_counts(pool: &ThreadPool, n_tasks: usize) -> Vec<usize> {
        let counts: Vec<AtomicUsize> = (0..n_tasks).map(|_| AtomicUsize::new(0)).collect();
        pool.run(n_tasks, |t| {
            counts[t].fetch_add(1, Ordering::Relaxed);
        });
        counts.into_iter().map(AtomicUsize::into_inner).collect()
    }

    #[test]
    fn every_task_runs_exactly_once() {
        for threads in [0, 1, 2, 3, 8] {
            let pool = ThreadPool::new(threads);
            assert_eq!(pool.threads(), threads.max(1));
            for n_tasks in [0, 1, 2, 7, 64, 1000] {
                assert_eq!(task_counts(&pool, n_tasks), vec![1; n_tasks]);
            }
        }
    }

    /// With no spinning, every job has to wake parked workers, which
    /// exercises the park handshake thousands of times.
    #[test]
    fn parked_workers_never_miss_a_job() {
        let pool = ThreadPool::with_spin(4, Duration::ZERO);
        for round in 0..2000 {
            assert_eq!(task_counts(&pool, 4 + round % 5), vec![1; 4 + round % 5]);
            if round % 100 == 0 {
                // Long enough for every worker to park.
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    #[test]
    fn task_panics_reach_the_caller_and_the_pool_survives() {
        let pool = ThreadPool::new(4);
        for bad in [0, 5, 63] {
            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                pool.run(64, |t| assert_ne!(t, bad, "task {t} fails"));
            }));
            let payload = result.expect_err("the panic must propagate");
            let message = payload.downcast_ref::<String>().unwrap();
            assert!(message.contains(&format!("task {bad} fails")), "{message}");
            assert_eq!(task_counts(&pool, 100), vec![1; 100]);
        }
    }

    #[test]
    fn nested_and_concurrent_runs_complete() {
        let pool = ThreadPool::new(4);
        let total = AtomicUsize::new(0);
        pool.run(8, |_| {
            pool.run(8, |_| {
                total.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(total.load(Ordering::Relaxed), 64);

        thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    for _ in 0..200 {
                        assert_eq!(task_counts(&pool, 16), vec![1; 16]);
                    }
                });
            }
        });
    }

    #[test]
    fn chunks_cover_every_element_once() {
        let pool = ThreadPool::new(3);
        let mut a = vec![0usize; 100];
        let mut b = vec![0usize; 7];
        let mut c: Vec<usize> = Vec::new();
        pool.for_each_chunk([&mut a[..], &mut b[..], &mut c[..]], 16, |s, chunk, out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v += 1000 * s + chunk * 16 + i + 1;
            }
        });
        assert_eq!(a, (1..=100).collect::<Vec<_>>());
        assert_eq!(b, (1001..=1007).collect::<Vec<_>>());

        let mut heads = vec![0u32; 4 * 8];
        let mut sums = vec![0u32; 4];
        pool.for_each_chunk_pair(&mut heads, 8, &mut sums, 1, |h, head, sum| {
            head.fill(h as u32);
            sum[0] = 8 * h as u32;
        });
        assert_eq!(sums, [0, 8, 16, 24]);
        assert!(
            heads
                .chunks(8)
                .enumerate()
                .all(|(h, c)| c.iter().all(|&v| v == h as u32))
        );
    }

    #[test]
    #[should_panic(expected = "do not split")]
    fn mismatched_chunk_pairs_are_rejected() {
        ThreadPool::new(2).for_each_chunk_pair(&mut [0; 8], 4, &mut [0; 3], 1, |_, _, _| {});
    }
}
