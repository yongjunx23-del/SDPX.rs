//! Bounded file conversion: scan the header, then feed one matrix per worker.
use crate::{require, write_block, write_output, PolynomialMatrix, Result};
use sdpx_arithmetic::Scalar;
use serde::de::{DeserializeSeed, Error, IgnoredAny, MapAccess, SeqAccess, Visitor};
use std::{
    fmt,
    fs::File,
    io::BufReader,
    path::Path,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::sync_channel,
        Arc, Mutex,
    },
};

#[derive(Debug, PartialEq)]
pub(crate) struct Header {
    pub objective: Vec<String>,
    pub normalization: Option<Vec<String>>,
    pub count: usize,
}

type Emit<'a> = Option<&'a mut dyn FnMut(PolynomialMatrix) -> Result<()>>;
struct Matrices<'a>(Emit<'a>);
impl<'de> DeserializeSeed<'de> for Matrices<'_> {
    type Value = usize;
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<usize, D::Error> {
        d.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for Matrices<'_> {
    type Value = usize;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a polynomial matrix array")
    }
    fn visit_seq<A: SeqAccess<'de>>(mut self, mut seq: A) -> std::result::Result<usize, A::Error> {
        let mut count = 0usize;
        if let Some(emit) = &mut self.0 {
            while let Some(matrix) = seq.next_element::<PolynomialMatrix>()? {
                emit(matrix).map_err(A::Error::custom)?;
                count = count
                    .checked_add(1)
                    .ok_or_else(|| A::Error::custom("too many matrices"))?;
            }
        } else {
            while seq.next_element::<IgnoredAny>()?.is_some() {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| A::Error::custom("too many matrices"))?;
            }
        }
        Ok(count)
    }
}
struct Program<'a>(Emit<'a>);
impl<'de> DeserializeSeed<'de> for Program<'_> {
    type Value = Header;
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Header, D::Error> {
        d.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for Program<'_> {
    type Value = Header;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a polynomial matrix program")
    }
    fn visit_map<A: MapAccess<'de>>(mut self, mut map: A) -> std::result::Result<Header, A::Error> {
        let (mut objective, mut normalization, mut count) = (None, None, None);
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "objective" => {
                    if objective.is_some() {
                        return Err(A::Error::duplicate_field("objective"));
                    }
                    objective = Some(map.next_value()?);
                }
                "normalization" => {
                    if normalization.is_some() {
                        return Err(A::Error::duplicate_field("normalization"));
                    }
                    normalization = Some(map.next_value()?);
                }
                "PositiveMatrixWithPrefactorArray" => {
                    if count.is_some() {
                        return Err(A::Error::duplicate_field(
                            "PositiveMatrixWithPrefactorArray",
                        ));
                    }
                    count = Some(map.next_value_seed(Matrices(self.0.take()))?);
                }
                _ => {
                    return Err(A::Error::unknown_field(
                        &key,
                        &[
                            "objective",
                            "normalization",
                            "PositiveMatrixWithPrefactorArray",
                        ],
                    ))
                }
            }
        }
        Ok(Header {
            objective: objective.ok_or_else(|| A::Error::missing_field("objective"))?,
            normalization: normalization.flatten(),
            count: count
                .ok_or_else(|| A::Error::missing_field("PositiveMatrixWithPrefactorArray"))?,
        })
    }
}
fn scan(path: &Path, emit: Emit<'_>) -> Result<Header> {
    let input = BufReader::new(File::open(path)?);
    match path.extension().and_then(|v| v.to_str()) {
        Some("xml") => crate::xml::scan(input, emit),
        Some("json") => {
            let mut d = serde_json::Deserializer::from_reader(input);
            let header = Program(emit).deserialize(&mut d)?;
            d.end()?;
            Ok(header)
        }
        _ => Err("PMP input must have a .json or .xml extension".into()),
    }
}

/// Convert a JSON or XML file without retaining all its polynomial matrices.
/// A header scan supports any field order. The second pass feeds at most one
/// matrix per worker plus the reader's current matrix. Output matches the
/// in-memory API; errors join workers before removing this call's directory.
/// Returns the number of converted blocks and workers. `threads == 0` uses
/// the available cores whose largest-block workspace fits in half the memory
/// allocation (physical RAM capped by cgroups and PBS). With `resample`, blocks given as
/// explicit sample points and scalings (PyCFTBoot/SDPB.m XML) take the SDPB
/// 3.1 points and bases of their recovered prefactor instead.
pub fn convert_file<T: Scalar + FromStr>(
    input: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    threads: usize,
    resample: bool,
) -> Result<(usize, usize)> {
    let (input, destination) = (input.as_ref(), destination.as_ref());
    let mut largest = 1;
    let mut estimate = |matrix| {
        largest = largest.max(workspace_bytes::<T>(&matrix, resample));
        Ok(())
    };
    let header = scan(
        input,
        if threads == 0 {
            Some(&mut estimate)
        } else {
            None
        },
    )?;
    let workers = if threads > 0 {
        threads
    } else {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        cores.min(
            (effective_memory() / 2 / largest)
                .max(1)
                .min(usize::MAX as u64) as usize,
        )
    }
    .min(header.count.max(1));
    write_output::<T>(
        &header.objective,
        header.normalization.as_deref(),
        header.count,
        destination,
        resample,
        |prepared| {
            if workers == 1 {
                let mut completed = Vec::with_capacity(header.count);
                let mut emit = |matrix| {
                    let (_, info) = write_block(&matrix, prepared, completed.len(), destination)?;
                    completed.push(info);
                    Ok(())
                };
                let actual = scan(input, Some(&mut emit))?;
                require(actual == header, "PMP header changed while reading input")?;
                return Ok(completed);
            }
            let failed = AtomicBool::new(false);
            std::thread::scope(|scope| -> Result<_> {
                // A rendezvous channel has no queued matrices: parsing overlaps
                // conversion, but slow workers apply backpressure to the reader.
                let (sender, receiver) = sync_channel::<(usize, PolynomialMatrix)>(0);
                let receiver = Arc::new(Mutex::new(receiver));
                let mut handles = Vec::with_capacity(workers);
                let mut spawn_error = None;
                for _ in 0..workers {
                    let receiver = Arc::clone(&receiver);
                    let failed = &failed;
                    let task = move || -> Result<Vec<_>> {
                        let mut completed = Vec::new();
                        loop {
                            let next = receiver.lock().map_err(|_| "input queue poisoned")?.recv();
                            let Ok((index, matrix)) = next else {
                                return Ok(completed);
                            };
                            match write_block(&matrix, prepared, index, destination) {
                                Ok(info) => completed.push(info),
                                Err(error) => {
                                    failed.store(true, Ordering::Relaxed);
                                    return Err(error);
                                }
                            }
                        }
                    };
                    match std::thread::Builder::new().spawn_scoped(scope, task) {
                        Ok(handle) => handles.push(handle),
                        Err(error) => {
                            spawn_error = Some(error);
                            break;
                        }
                    }
                }
                // Only workers own the receiver. If they all fail or panic,
                // its last owner closes the channel and wakes a blocked send.
                drop(receiver);
                let parsed = if spawn_error.is_none() {
                    let mut index = 0usize;
                    let mut emit = |matrix| {
                        require(!failed.load(Ordering::Relaxed), "conversion worker failed")?;
                        sender
                            .send((index, matrix))
                            .map_err(|_| "conversion workers stopped")?;
                        index += 1;
                        Ok(())
                    };
                    scan(input, Some(&mut emit))
                } else {
                    Err("cannot start conversion workers".into())
                };
                drop(sender);
                let results: Vec<_> = handles
                    .into_iter()
                    .map(|h| {
                        h.join()
                            .unwrap_or_else(|_| Err("conversion worker panicked".into()))
                    })
                    .collect();
                if let Some(error) = spawn_error {
                    return Err(error.into());
                }
                let mut completed: Vec<_> = results
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                require(parsed? == header, "PMP header changed while reading input")?;
                completed.sort_unstable_by_key(|(index, _)| *index);
                Ok(completed.into_iter().map(|(_, info)| info).collect())
            })
        },
    )?;
    Ok((header.count, workers))
}

// Input strings and vectors remain live while conversion builds both basis
// parities and the adjusted polynomial vector for one matrix entry. Recovery
// also keeps derivative polynomials at 2048 bits. Budget the largest block,
// not a mean that lets many small blocks hide one large worker allocation.
fn workspace_bytes<T: Scalar>(m: &PolynomialMatrix, resample: bool) -> u64 {
    let mut bytes = 0u64;
    let mut add = |s: &String| bytes = bytes.saturating_add(s.capacity() as u64 + 128);
    let mut count = 1usize;
    let mut variables = 0usize;
    for vector in m.polynomials.iter().flatten() {
        variables = variables.max(vector.len());
        for poly in vector {
            count = count.max(poly.len());
            poly.iter().for_each(&mut add);
        }
    }
    for p in [m.prefactor.as_ref(), m.reduced_prefactor.as_ref()]
        .into_iter()
        .flatten()
    {
        add(&p.constant);
        add(&p.base);
        p.poles.iter().for_each(&mut add);
    }
    for v in [
        &m.sample_points,
        &m.sample_scalings,
        &m.reduced_sample_scalings,
    ]
    .into_iter()
    .flatten()
    {
        v.iter().for_each(&mut add);
    }
    for basis in [
        &m.bilinear_basis,
        &m.bilinear_basis_even,
        &m.bilinear_basis_odd,
    ]
    .into_iter()
    .flatten()
    {
        basis.iter().flatten().for_each(&mut add);
    }
    count = count.max(m.sample_points.as_ref().map_or(0, Vec::len));
    count = count.saturating_add(m.reduced_prefactor.as_ref().map_or(0, |p| p.poles.len()));
    let scalar = if resample && m.prefactor.is_none() && m.sample_points.is_some() {
        std::mem::size_of::<T>().max(std::mem::size_of::<sdpx_arithmetic::MpFloat<32>>())
    } else {
        std::mem::size_of::<T>()
    } as u64;
    let n = count as u64;
    bytes
        .saturating_mul(2)
        .saturating_add(
            scalar.saturating_mul(
                n.saturating_mul(n)
                    .saturating_mul(4)
                    .saturating_add(n.saturating_mul(variables as u64).saturating_mul(4))
                    .saturating_add(n.saturating_mul(16)),
            ),
        )
        .saturating_add(16 << 20)
}

fn memory_size(text: &str) -> Option<u64> {
    let text = text.trim().to_ascii_lowercase();
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    let n: u64 = text[..end].parse().ok()?;
    let shift = match text[end..].trim() {
        "" | "b" => 0,
        "kb" | "kib" => 10,
        "mb" | "mib" => 20,
        "gb" | "gib" => 30,
        "tb" | "tib" => 40,
        _ => return None,
    };
    n.checked_mul(1 << shift)
}

fn effective_memory() -> u64 {
    let mut memory = physical_memory();
    #[cfg(target_os = "linux")]
    if let Some(limit) = cgroup_memory_limit() {
        memory = memory.min(limit);
    }
    let pbs = ["PBS_RESC_MEM", "PBS_MEM"]
        .into_iter()
        .filter_map(|key| std::env::var(key).ok().and_then(|s| memory_size(&s)))
        .min();
    let pbs = pbs.or_else(|| {
        let job = std::env::var("PBS_JOBID").ok()?;
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};
        let mut child = Command::new("qstat")
            .args(["-f", &job])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(20))
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
            }
        }
        let out = child.wait_with_output().ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8(out.stdout)
            .ok()?
            .lines()
            .filter_map(|line| {
                let (key, value) = line.trim().split_once('=')?;
                matches!(key.trim(), "Resource_List.mem" | "Resource_List.vmem")
                    .then(|| memory_size(value))
                    .flatten()
            })
            .min()
    });
    if let Some(limit) = pbs {
        memory = memory.min(limit);
    } else if std::env::var_os("PBS_JOBID").is_some() {
        // An unenforced PBS allocation may be smaller than host RAM; when
        // qstat is unavailable, auto mode conservatively uses one worker.
        memory = 0;
    }
    memory
}

#[cfg(target_os = "linux")]
fn cgroup_memory_limit() -> Option<u64> {
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mounts = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    let decode = |s: &str| {
        s.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
    };
    let mut limit: Option<u64> = None;
    for group in groups.lines() {
        let mut fields = group.splitn(3, ':');
        fields.next();
        let (controllers, group) = (fields.next()?, fields.next()?);
        let (kind, file) = if controllers.is_empty() {
            ("cgroup2", "memory.max")
        } else if controllers.split(',').any(|c| c == "memory") {
            ("cgroup", "memory.limit_in_bytes")
        } else {
            continue;
        };
        for mount in mounts.lines() {
            let Some((fields, options)) = mount.split_once(" - ") else {
                continue;
            };
            let mut options = options.split_whitespace();
            if options.next() != Some(kind) {
                continue;
            }
            options.next();
            if kind == "cgroup" && !options.next()?.split(',').any(|c| c == "memory") {
                continue;
            }
            let mut fields = fields.split_whitespace();
            let root = decode(fields.nth(3)?);
            let mount = decode(fields.next()?);
            let mount = Path::new(&mount);
            let Ok(relative) = Path::new(group).strip_prefix(&root) else {
                continue;
            };
            let mut path = mount.join(relative);
            loop {
                if let Some(bytes) = std::fs::read_to_string(path.join(file))
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                {
                    limit = Some(limit.map_or(bytes, |old| old.min(bytes)));
                }
                if path == mount || !path.pop() {
                    break;
                }
            }
        }
    }
    limit
}

/// Physical RAM in bytes.
fn physical_memory() -> u64 {
    extern "C" {
        fn sysconf(name: std::ffi::c_int) -> std::ffi::c_long;
    }
    #[cfg(target_os = "macos")]
    const NAMES: [std::ffi::c_int; 2] = [200, 29]; // _SC_PHYS_PAGES, _SC_PAGESIZE
    #[cfg(not(target_os = "macos"))]
    const NAMES: [std::ffi::c_int; 2] = [85, 30];
    // SAFETY: sysconf reads a system constant.
    let [pages, size] = NAMES.map(|name| unsafe { sysconf(name) }.max(0) as u64);
    pages * size
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdpx_arithmetic::{Bits1024, Bits128};

    #[test]
    fn worker_workspace_tracks_large_blocks_and_precision() {
        let mut m = PolynomialMatrix::default();
        m.polynomials = vec![vec![vec![vec!["1".into(); 1000]]]];
        let low = workspace_bytes::<Bits128>(&m, false);
        let high = workspace_bytes::<Bits1024>(&m, false);
        assert!(high > low * 2);
        let small = workspace_bytes::<Bits128>(&PolynomialMatrix::default(), false);
        assert!(low > small * 4);
        assert_eq!(memory_size("32gb"), Some(32 << 30));
        assert_eq!(memory_size("1048576"), Some(1 << 20));
    }
}
