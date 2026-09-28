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
/// Returns the number of converted blocks.
pub fn convert_file<T: Scalar + FromStr>(
    input: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    threads: usize,
) -> Result<usize> {
    require(threads > 0, "threads must be positive")?;
    let (input, destination) = (input.as_ref(), destination.as_ref());
    let header = scan(input, None)?;
    write_output::<T>(
        &header.objective,
        header.normalization.as_deref(),
        header.count,
        destination,
        |prepared| {
            let workers = threads.min(header.count);
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
    Ok(header.count)
}
