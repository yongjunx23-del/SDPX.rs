//! Read SDPB's uncompressed `pmp2sdp --outputFormat=json` output directly.
//! This is an input adapter; direction, scaling and solves use the shared core.
use super::*;
use crate::algebra::*;
use crate::solver::SupportedConeT;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{fs::File, io::BufReader, path::Path, str::FromStr};

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
#[serde(deny_unknown_fields)]
struct Objectives {
    constant: String,
    b: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockInfo {
    dim: usize,
    num_points: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockData {
    bilinear_bases_even: Vec<Vec<String>>,
    bilinear_bases_odd: Vec<Vec<String>>,
    c: Vec<String>,
    #[serde(rename = "B")]
    b: Vec<Vec<String>>,
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
    let objective: Objectives = read(directory, "objectives.json")?;
    let objective_constant = decimal(&objective.constant)?;
    let mut rhs: Vec<T> = objective
        .b
        .iter()
        .map(|s| decimal(s))
        .collect::<Result<_, _>>()?;
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
    for block_index in 0..control.num_blocks {
        let info: BlockInfo = read(directory, &format!("block_info_{block_index}.json"))?;
        let data: BlockData = read(directory, &format!("block_data_{block_index}.json"))?;
        let count = triangle(info.dim)?
            .checked_mul(info.num_points)
            .ok_or_else(|| bad("sampled column count overflow"))?;
        if info.dim == 0 || info.num_points == 0 || data.c.len() != count || data.b.len() != count {
            return Err(bad("inconsistent sampled block dimensions"));
        }
        let column_start = q.len();
        for (c, b) in data.c.iter().zip(&data.b) {
            if b.len() != equalities {
                return Err(bad("sampled B width mismatch"));
            }
            q.push(decimal(c)?);
            for (row, text) in b.iter().enumerate() {
                let value: T = decimal(text)?;
                if !value.is_zero() {
                    rowval.push(row);
                    nzval.push(value);
                }
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
                    values.push(decimal(&row[k])?);
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
