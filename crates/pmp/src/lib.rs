//! Polynomial matrix programs to SDPB sampled SDP directories.
//!
//! Adapted from SDPB 3.1.0 pmp2sdp (MIT); see provenance/SDPB-LICENSE.
//! Decimal strings are parsed directly at the caller's MPFR precision. No
//! solver, MPI installation, Julia runtime, or BLAS provider is required.
mod sampling;
mod stream;
pub use stream::convert_file;
mod xml;
pub use sampling::Prefactor;
use sampling::{evaluate, Damped};
use sdpx_arithmetic::Scalar;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    str::FromStr,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn decimal<T: Scalar + FromStr>(s: &str) -> Result<T> {
    // Check decimal syntax without any floating-point intermediary.
    let text = s.trim();
    let mantissa = text.split(['e', 'E']).next().unwrap_or("");
    require(
        !text.is_empty()
            && mantissa.chars().any(|c| c.is_ascii_digit())
            && text
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E')),
        "expected a decimal string",
    )?;
    let v: T = text
        .parse()
        .map_err(|_| format!("invalid decimal: {text}"))?;
    require(v.is_finite(), "decimal overflows working precision")?;
    require(
        v != T::zero() || !mantissa.chars().any(|c| matches!(c, '1'..='9')),
        "nonzero decimal underflows working precision",
    )?;
    Ok(v)
}
fn numbers<T: Scalar + FromStr>(values: &[String]) -> Result<Vec<T>> {
    values.iter().map(|s| decimal(s)).collect()
}

fn strings<T: Scalar>(values: &[T]) -> Vec<String> {
    values.iter().map(Scalar::decimal_string).collect()
}
type Polynomials = Vec<Vec<Vec<Vec<String>>>>;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialMatrixProgram {
    pub objective: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization: Option<Vec<String>>,
    #[serde(rename = "PositiveMatrixWithPrefactorArray")]
    pub matrices: Vec<PolynomialMatrix>,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct PolynomialMatrix {
    pub polynomials: Polynomials,
    #[serde(default, alias = "DampedRational")]
    pub prefactor: Option<Prefactor>,
    #[serde(default)]
    pub reduced_prefactor: Option<Prefactor>,
    #[serde(default)]
    pub sample_points: Option<Vec<String>>,
    #[serde(default)]
    pub sample_scalings: Option<Vec<String>>,
    #[serde(default)]
    pub reduced_sample_scalings: Option<Vec<String>>,
    #[serde(default)]
    pub bilinear_basis: Option<Vec<Vec<String>>>,
    #[serde(default, rename = "bilinearBasis_0")]
    pub bilinear_basis_even: Option<Vec<Vec<String>>>,
    #[serde(default, rename = "bilinearBasis_1")]
    pub bilinear_basis_odd: Option<Vec<Vec<String>>>,
}
impl PolynomialMatrixProgram {
    /// Read SDPB JSON or legacy XML. JSON numerical fields must be strings.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match path.extension().and_then(|v| v.to_str()) {
            Some("json") => Ok(serde_json::from_reader(std::io::BufReader::new(
                fs::File::open(path)?,
            ))?),
            Some("xml") => xml::read(std::io::BufReader::new(fs::File::open(path)?)),
            _ => Err("PMP input must have a .json or .xml extension".into()),
        }
    }

    /// Convert at T's fixed precision into a new SDPB JSON directory.
    /// Existing paths are refused. Blocks are converted and written one at a
    /// time, without constructing the expanded SDP coefficient matrix.
    pub fn write_sdp<T: Scalar + FromStr>(&self, destination: impl AsRef<Path>) -> Result<()> {
        self.write_sdp_with_threads::<T>(destination, 1)
    }

    /// Convert independent blocks with at most `threads` workers, writing each
    /// output row immediately. Files and block order equal serial output.
    /// The output directory is removed on failure after all workers join.
    pub fn write_sdp_with_threads<T: Scalar + FromStr>(
        &self,
        destination: impl AsRef<Path>,
        threads: usize,
    ) -> Result<()> {
        require(threads > 0, "threads must be positive")?;
        write_output::<T>(
            &self.objective,
            self.normalization.as_deref(),
            self.matrices.len(),
            destination.as_ref(),
            |prepared| {
                let destination = destination.as_ref();
                let write_block = |index: usize| -> Result<_> {
                    write_block(&self.matrices[index], prepared, index, destination)
                };
                let workers = threads.min(self.matrices.len());
                let mut completed = if workers == 1 {
                    (0..self.matrices.len())
                        .map(write_block)
                        .collect::<Result<Vec<_>>>()?
                } else {
                    // Blocks have different sizes. Claim the next block only after
                    // writing the current one, bounding live buffers by workers.
                    let next = std::sync::atomic::AtomicUsize::new(0);
                    std::thread::scope(|scope| -> Result<Vec<_>> {
                        let mut handles = Vec::with_capacity(workers);
                        let mut spawn_error = None;
                        for _ in 0..workers {
                            let task = || -> Result<Vec<_>> {
                                let mut completed = Vec::new();
                                loop {
                                    let index =
                                        next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if index >= self.matrices.len() {
                                        return Ok(completed);
                                    }
                                    completed.push(write_block(index)?);
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
                        // Join every worker before propagating errors or cleaning up.
                        let results: Vec<_> = handles
                            .into_iter()
                            .map(|handle| {
                                handle
                                    .join()
                                    .unwrap_or_else(|_| Err("conversion worker panicked".into()))
                            })
                            .collect();
                        if let Some(error) = spawn_error {
                            return Err(error.into());
                        }
                        Ok(results
                            .into_iter()
                            .collect::<Result<Vec<_>>>()?
                            .into_iter()
                            .flatten()
                            .collect())
                    })?
                };
                completed.sort_unstable_by_key(|(index, _)| *index);
                Ok(completed.into_iter().map(|(_, info)| info).collect())
            },
        )
    }
}

struct Prepared<T> {
    norm: Vec<T>,
    pivot: usize,
}

fn write_block<T: Scalar + FromStr>(
    matrix: &PolynomialMatrix,
    prepared: &Prepared<T>,
    index: usize,
    destination: &Path,
) -> Result<(usize, serde_json::Value)> {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination.join(format!("block_data_{index}.json")))?;
    let mut output = BufWriter::new(file);
    let (count, info) = convert::<T>(matrix, &prepared.norm, prepared.pivot, index, &mut output)
        .map_err(|e| format!("block {index}: {e}"))?;
    output.flush()?;
    write_json(
        destination,
        &format!("block_info_{index}.json"),
        &serde_json::json!({"dim":info["dim"],"num_points":count}),
    )?;
    Ok((index, info))
}

fn write_output<T: Scalar + FromStr>(
    objective: &[String],
    normalization: Option<&[String]>,
    count: usize,
    destination: &Path,
    blocks: impl FnOnce(&Prepared<T>) -> Result<Vec<serde_json::Value>>,
) -> Result<()> {
    require(
        T::precision_bits() >= 128,
        "PMP conversion requires MPFR precision of at least 128 bits",
    )?;
    require(count > 0, "PMP must contain at least one matrix")?;
    let objective = numbers::<T>(objective)?;
    require(
        !objective.is_empty(),
        "objective must contain a constant coordinate",
    )?;
    let norm = match normalization {
        Some(v) => numbers::<T>(v)?,
        None => {
            let mut v = vec![T::zero(); objective.len()];
            v[0] = T::one();
            v
        }
    };
    require(
        norm.len() == objective.len(),
        "normalization and objective lengths differ",
    )?;
    let mut pivot = 0;
    for i in 1..norm.len() {
        if norm[i].abs() > norm[pivot].abs() {
            pivot = i;
        }
    }
    require(norm[pivot] != T::zero(), "normalization must be nonzero")?;
    let constant = objective[pivot] / norm[pivot];
    let b: Vec<_> = objective
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != pivot)
        .map(|(i, &v)| v - norm[i] * constant)
        .collect();
    require(
        constant.is_finite() && b.iter().all(|v| v.is_finite()),
        "normalized objective overflow",
    )?;
    require(!destination.exists(), "output path already exists")?;
    fs::create_dir(destination)?;
    let result = (|| {
        write_json(
            destination,
            "objectives.json",
            &serde_json::json!({"constant":constant.decimal_string(),"b":strings(&b)}),
        )?;
        let prepared = Prepared { norm, pivot };
        let metadata = blocks(&prepared)?;
        require(
            metadata.len() == count,
            "matrix count changed while reading input",
        )?;
        if normalization.is_some() {
            write_json(
                destination,
                "normalization.json",
                &serde_json::json!({"normalization":strings(&prepared.norm)}),
            )?;
        }
        write_json(destination, "pmp_info.json", &metadata)?;
        // Publish control last, only after all workers and input validation finish.
        write_json(
            destination,
            "control.json",
            &serde_json::json!({"num_blocks":count,"command":format!("sdpx-pmp2sdp --precision {}", T::precision_bits())}),
        )
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(destination);
    }
    result
}

fn write_json(path: &Path, name: &str, value: &impl Serialize) -> Result<()> {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path.join(name))?;
    let mut output = BufWriter::new(file);
    serde_json::to_writer(&mut output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}
/// Append one JSON array element without retaining earlier rows.
fn write_element(output: &mut impl Write, first: &mut bool, value: &impl Serialize) -> Result<()> {
    if !*first {
        output.write_all(b",")?;
    }
    *first = false;
    serde_json::to_writer(output, value)?;
    Ok(())
}
fn convert<T: Scalar + FromStr>(
    m: &PolynomialMatrix,
    norm: &[T],
    pivot: usize,
    index: usize,
    output: &mut impl Write,
) -> Result<(usize, serde_json::Value)> {
    let dim = m.polynomials.len();
    require(
        dim > 0 && m.polynomials.iter().all(|row| row.len() == dim),
        "polynomial matrix must be nonempty and square",
    )?;
    let mut degree = 0;
    for row in &m.polynomials {
        for vector in row {
            require(
                vector.len() == norm.len(),
                "polynomial vector and objective lengths differ",
            )?;
            for poly in vector {
                require(
                    !poly.is_empty(),
                    "polynomial coefficient list cannot be empty",
                )?;
                degree = degree.max(poly.len() - 1);
            }
        }
    }
    for r in 0..dim {
        for c in 0..r {
            // Equal strings need parsing only once, when evaluating the upper
            // triangle. Otherwise preserve numeric (not textual) symmetry.
            let (a, b) = (&m.polynomials[r][c], &m.polynomials[c][r]);
            if a != b {
                for (a, b) in a.iter().zip(b) {
                    require(a.len() == b.len(), "polynomial matrix must be symmetric")?;
                    for (a, b) in a.iter().zip(b) {
                        require(
                            decimal::<T>(a)? == decimal::<T>(b)?,
                            "polynomial matrix must be symmetric",
                        )?;
                    }
                }
            }
        }
    }
    let prefactor = Damped::<T>::read(m.prefactor.as_ref(), degree)?;
    require(
        m.reduced_prefactor.is_none() || m.prefactor.is_some(),
        "reducedPrefactor requires prefactor",
    )?;
    let reduced = m
        .reduced_prefactor
        .as_ref()
        .map(|p| Damped::read(Some(p), degree))
        .transpose()?;
    let reduced = reduced.as_ref().unwrap_or(&prefactor);
    let count = (degree + 1)
        .checked_add(reduced.poles.len())
        .and_then(|n| n.checked_sub(prefactor.poles.len()))
        .ok_or("invalid reduced sample count")?;
    require(count > 0, "reduced sample count must be positive")?;
    let points = match &m.sample_points {
        Some(v) => numbers(v)?,
        None => sampling::points(count, &reduced)?,
    };
    require(
        points.len() == count
            && points.iter().all(|&x| x >= T::zero())
            && points.windows(2).all(|w| w[0] < w[1]),
        "sample points must be nonnegative, increasing and match the polynomial degree",
    )?;
    let scales = match &m.sample_scalings {
        Some(v) => numbers(v)?,
        None => prefactor.scalings(&points)?,
    };
    let reduced_scalings = match &m.reduced_sample_scalings {
        Some(v) => Some(numbers(v)?),
        None if m.reduced_prefactor.is_some() => Some(reduced.scalings(&points)?),
        None => None,
    };
    let rscales = reduced_scalings.as_deref().unwrap_or(&scales);
    require(
        [scales.as_slice(), rscales]
            .iter()
            .all(|s| s.len() == count && s.iter().all(|&v| v > T::zero())),
        "sample scalings must be positive and match sample points",
    )?;
    let supplied = [
        m.bilinear_basis_even.as_ref().or(m.bilinear_basis.as_ref()),
        m.bilinear_basis_odd.as_ref().or(m.bilinear_basis.as_ref()),
    ];
    let effective_degree = count - 1;
    let sizes = [effective_degree / 2 + 1, (effective_degree + 1) / 2];
    let basis = if supplied.iter().all(Option::is_none) {
        sampling::basis(&points, rscales)?
    } else {
        let mut result = [Vec::new(), Vec::new()];
        for parity in 0..2 {
            if sizes[parity] == 0 {
                continue;
            }
            let input = supplied[parity].ok_or("both bilinear basis parities must be supplied")?;
            require(
                input.len() >= sizes[parity],
                "bilinear basis has too few polynomials",
            )?;
            for (j, poly) in input.iter().take(sizes[parity]).enumerate() {
                let mut coeff = numbers::<T>(poly)?;
                while coeff.len() > j + 1 && coeff.last() == Some(&T::zero()) {
                    coeff.pop();
                }
                require(
                    coeff.len() == j + 1 && coeff[j] != T::zero(),
                    "bilinear polynomial j must have degree j",
                )?;
                result[parity].push(coeff);
            }
        }
        result
    };
    output.write_all(b"{\"bilinear_bases_even\":[")?;
    for parity in 0..2 {
        if parity == 1 {
            output.write_all(b"],\"bilinear_bases_odd\":[")?;
        }
        let mut first = true;
        let roots: Vec<_> = points
            .iter()
            .enumerate()
            .map(|(k, &x)| {
                let s = if parity == 0 {
                    rscales[k]
                } else {
                    rscales[k] * x
                };
                s.sqrt()
            })
            .collect();
        for poly in &basis[parity] {
            let mut row = Vec::with_capacity(count);
            for (k, &x) in points.iter().enumerate() {
                let value = roots[k] * evaluate(poly, x);
                require(value.is_finite(), "sampled basis overflow")?;
                row.push(value.decimal_string());
            }
            write_element(output, &mut first, &row)?;
        }
    }
    drop(basis);
    output.write_all(b"],\"c\":[")?;
    let inverse = T::one() / norm[pivot];
    let mut first = true;
    for c in 0..dim {
        for row in m.polynomials.iter().take(c + 1) {
            let cp: Vec<_> = numbers::<T>(&row[c][pivot])?
                .into_iter()
                .map(|a| a * inverse)
                .collect();
            for (k, &x) in points.iter().enumerate() {
                let value = scales[k] * evaluate(&cp, x);
                require(value.is_finite(), "constraint constant overflow")?;
                write_element(output, &mut first, &value.decimal_string())?;
            }
        }
    }
    output.write_all(b"],\"B\":[")?;
    let mut first = true;
    let mut values = Vec::with_capacity(norm.len() - 1);
    for c in 0..dim {
        for row in m.polynomials.iter().take(c + 1) {
            let v = &row[c];
            let cp: Vec<_> = numbers::<T>(&v[pivot])?
                .into_iter()
                .map(|a| a * inverse)
                .collect();
            let mut adjusted = Vec::with_capacity(norm.len() - 1);
            for i in 0..norm.len() {
                if i != pivot {
                    let mut p = numbers::<T>(&v[i])?;
                    p.resize(p.len().max(cp.len()), T::zero());
                    for (a, &b) in p.iter_mut().zip(&cp) {
                        *a -= norm[i] * b;
                    }
                    adjusted.push(p);
                }
            }
            for (k, &x) in points.iter().enumerate() {
                values.clear();
                for p in &adjusted {
                    let value = -scales[k] * evaluate(p, x);
                    require(value.is_finite(), "constraint coefficient overflow")?;
                    values.push(value.decimal_string());
                }
                write_element(output, &mut first, &values)?;
            }
        }
    }
    let metadata = serde_json::json!({"index":index,"path":"","dim":dim,"prefactor":prefactor.metadata(),"reducedPrefactor":reduced.metadata(),"samplePoints":strings(&points),"sampleScalings":strings(&scales),"reducedSampleScalings":strings(rscales)});
    output.write_all(b"]}\n")?;
    Ok((count, metadata))
}
