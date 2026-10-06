//! Read SDPB's uncompressed `pmp2sdp --outputFormat=json` output directly.
//! This is an input adapter; direction, scaling and solves use the shared core.
use super::*;
use crate::solver::SupportedConeT;
use serde::{
    de::{DeserializeOwned, Error, SeqAccess, Visitor},
    Deserialize, Serialize,
};
use std::{fmt, fs::File, io::BufReader, marker::PhantomData, path::Path, str::FromStr};

/// Original SDPB matrix location in the conic slack and dual vectors.
#[derive(Debug, Serialize)]
pub struct SampledGramLayout {
    /// Zero-based original block index.
    pub block_index: usize,
    /// Even (0) or odd (1) basis.
    pub parity: usize,
    /// Zero-based first svec row.
    pub row_start: usize,
    /// PSD matrix order. Off-diagonal svec entries include sqrt(2).
    pub side: usize,
}

/// Primal sampled formulation: min c'lambda, B'lambda=b,
/// X_j=sum_p lambda_p A_jp positive semidefinite.
pub struct SampledJsonProblem<T: FloatT> {
    /// Shared conic input: x=lambda; PSD s=svec(X), z=svec(Y).
    pub problem: JsonProblem<T>,
    /// Add to either conic objective to recover the original SDPB objective.
    pub objective_constant: T,
    /// y=-z[..num_equalities] in the original SDPB convention.
    pub num_equalities: usize,
    /// Matrix recovery map in input block/parity order.
    pub grams: Vec<SampledGramLayout>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Control {
    num_blocks: usize,
    #[serde(default, rename = "command")]
    _command: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields, bound(deserialize = "T: FloatT + FromStr"))]
struct Objectives<T> {
    constant: Decimal<T>,
    b: Vec<Decimal<T>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockInfo {
    dim: usize,
    num_points: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields, bound(deserialize = "T: FloatT + FromStr"))]
struct BlockData<T> {
    bilinear_bases_even: Vec<Vec<Decimal<T>>>,
    bilinear_bases_odd: Vec<Vec<Decimal<T>>>,
    c: Vec<Decimal<T>>,
    #[serde(rename = "B")]
    b: Vec<SparseRow<T>>,
}

// Parse transient JSON string slices directly at the working precision.
// Deserializing T itself would also admit JSON numbers for some backends.
struct Decimal<T>(T);
impl<'de, T: FloatT + FromStr> Deserialize<'de> for Decimal<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct DecimalVisitor<T>(PhantomData<T>);
        impl<'de, T: FloatT + FromStr> Visitor<'de> for DecimalVisitor<T> {
            type Value = Decimal<T>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a finite decimal string")
            }
            fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
                decimal(value).map(Decimal).map_err(E::custom)
            }
        }
        d.deserialize_str(DecimalVisitor(PhantomData))
    }
}

// B rows become CSC columns. Drop zeros while reading, but retain the full
// width so sparse storage cannot hide a malformed input row.
struct SparseRow<T> {
    width: usize,
    entries: Vec<(usize, T)>,
}
impl<'de, T: FloatT + FromStr> Deserialize<'de> for SparseRow<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct RowVisitor<T>(PhantomData<T>);
        impl<'de, T: FloatT + FromStr> Visitor<'de> for RowVisitor<T> {
            type Value = SparseRow<T>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an array of decimal strings")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut row = SparseRow {
                    width: 0,
                    entries: Vec::new(),
                };
                while let Some(Decimal(value)) = seq.next_element::<Decimal<T>>()? {
                    if !value.is_zero() {
                        row.entries.push((row.width, value));
                    }
                    row.width = row
                        .width
                        .checked_add(1)
                        .ok_or_else(|| A::Error::custom("sampled B width overflow"))?;
                }
                Ok(row)
            }
        }
        d.deserialize_seq(RowVisitor(PhantomData))
    }
}

fn read<D: DeserializeOwned>(directory: &Path, name: &str) -> Result<D, SolverError> {
    Ok(serde_json::from_reader(BufReader::new(File::open(
        directory.join(name),
    )?))?)
}

fn bad(message: &str) -> SolverError {
    SolverError::SampledInput(message.to_owned())
}

fn decimal<T: FloatT + FromStr>(s: &str) -> Result<T, SolverError> {
    // Restrict the wire format to decimal, excluding MPFR's special values and
    // alternative exponent syntax. Parsing at T's precision never uses f64.
    let value = s
        .parse::<T>()
        .map_err(|_| bad("invalid decimal coefficient"))?;
    if s.is_empty()
        || !s
            .bytes()
            .all(|c| c.is_ascii_digit() || matches!(c, b'+' | b'-' | b'.' | b'e' | b'E'))
        || !value.is_finite()
    {
        return Err(bad("sampled coefficients must be finite decimal strings"));
    }
    let mantissa = s.split(['e', 'E']).next().unwrap_or("");
    if value.is_zero() && mantissa.bytes().any(|c| matches!(c, b'1'..=b'9')) {
        return Err(bad("sampled coefficient underflow"));
    }
    Ok(value)
}

fn triangle(n: usize) -> Result<usize, SolverError> {
    n.checked_add(1)
        .and_then(|v| n.checked_mul(v))
        .map(|v| v / 2)
        .ok_or_else(|| bad("sampled dimensions overflow"))
}

/// Load sampled factors and ordinary B coefficients, without Julia or dense
/// PSD coefficient materialization. Normalization metadata has already been
/// applied by pmp2sdp and is not applied a second time here.
pub fn read_sdpb_sampled<T>(
    directory: impl AsRef<Path>,
) -> Result<SampledJsonProblem<T>, SolverError>
where
    T: FloatT + Serialize + DeserializeOwned + FromStr,
{
    let directory = directory.as_ref();
    let control: Control = read(directory, "control.json")?;
    // Match the sampled JSON schema, allowing the two producer metadata files
    // without reapplying their already-consumed normalization.
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".json") {
            continue;
        }
        let known = matches!(
            name.as_ref(),
            "control.json" | "objectives.json" | "normalization.json" | "pmp_info.json"
        ) || ["block_info_", "block_data_"].iter().any(|prefix| {
            name.strip_prefix(prefix)
                .and_then(|s| s.strip_suffix(".json"))
                .and_then(|s| s.parse::<usize>().ok().map(|n| (s, n)))
                .is_some_and(|(s, n)| n < control.num_blocks && s == n.to_string())
        });
        if !known {
            return Err(bad(&format!("unexpected sampled JSON file: {name}")));
        }
    }
    let objective: Objectives<T> = read(directory, "objectives.json")?;
    let objective_constant = objective.constant.0;
    let mut rhs: Vec<T> = objective.b.into_iter().map(|v| v.0).collect();
    let equalities = rhs.len();
    let mut cones = Vec::new();
    if equalities > 0 {
        cones.push(SupportedConeT::ZeroConeT(equalities));
    }
    let mut q = Vec::new();
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    let mut sampled = Vec::new();
    let mut grams = Vec::new();
    let mut rows = equalities;
    // Parsing wide decimals dominates loading; parse a batch of block files
    // in parallel, then append them in block order (identical result). The
    // batch bounds how many parsed-but-unappended blocks are alive at once.
    let batch = rayon::current_num_threads().max(1);
    let mut parsed = Vec::new();
    for block_index in 0..control.num_blocks {
        if block_index % batch == 0 {
            use rayon::prelude::*;
            let end = (block_index + batch).min(control.num_blocks);
            parsed = (block_index..end)
                .into_par_iter()
                .map(|i| -> Result<(BlockInfo, BlockData<T>), SolverError> {
                    Ok((
                        read(directory, &format!("block_info_{i}.json"))?,
                        read(directory, &format!("block_data_{i}.json"))?,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            parsed.reverse();
        }
        let (info, data) = parsed.pop().unwrap();
        let count = triangle(info.dim)?
            .checked_mul(info.num_points)
            .ok_or_else(|| bad("sampled column count overflow"))?;
        if info.dim == 0 || info.num_points == 0 || data.c.len() != count || data.b.len() != count {
            return Err(bad("inconsistent sampled block dimensions"));
        }
        let column_start = q.len();
        for (Decimal(c), b) in data.c.into_iter().zip(data.b) {
            if b.width != equalities {
                return Err(bad("sampled B width mismatch"));
            }
            q.push(c);
            for (row, value) in b.entries {
                rowval.push(row);
                nzval.push(value);
            }
            colptr.push(nzval.len());
        }
        for (parity, basis) in [data.bilinear_bases_even, data.bilinear_bases_odd]
            .into_iter()
            .enumerate()
        {
            if basis.is_empty() {
                continue;
            }
            let height = basis.len();
            if basis.iter().any(|r| r.len() != info.num_points) {
                return Err(bad("sampled basis width mismatch"));
            }
            let side = info
                .dim
                .checked_mul(height)
                .ok_or_else(|| bad("sampled PSD order overflow"))?;
            let end = rows
                .checked_add(triangle(side)?)
                .ok_or_else(|| bad("sampled row count overflow"))?;
            let size = height
                .checked_mul(info.num_points)
                .ok_or_else(|| bad("sampled basis size overflow"))?;
            let mut values = Vec::with_capacity(size);
            for k in 0..info.num_points {
                for row in &basis {
                    values.push(row[k].0);
                }
            }
            sampled.push(SampledBlock {
                row_start: rows,
                column_start,
                dim: info.dim,
                basis_rows: height,
                basis_cols: info.num_points,
                basis: values,
                weights: vec![-T::one(); count],
            });
            grams.push(SampledGramLayout {
                block_index,
                parity,
                row_start: rows,
                side,
            });
            cones.push(SupportedConeT::PSDTriangleConeT(side));
            rows = end;
        }
    }
    rhs.resize(rows, T::zero());
    let n = q.len();
    Ok(SampledJsonProblem {
        problem: JsonProblem {
            P: CscMatrix::zeros((n, n)),
            q,
            A: CscMatrix {
                m: rows,
                n,
                colptr,
                rowval,
                nzval,
            },
            b: rhs,
            cones,
            settings: DefaultSettings::default(),
            sampled,
        },
        objective_constant,
        num_equalities: equalities,
        grams,
    })
}
