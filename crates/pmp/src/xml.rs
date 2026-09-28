//! Streaming reader for pmp2sdp's legacy XML schema.
use crate::{require, PolynomialMatrix, PolynomialMatrixProgram, Result};
use quick_xml::{events::Event, Reader};
use std::io::BufRead;

enum Token {
    Open(String),
    Close,
    Text(String),
    Eof,
}
struct Xml<R> {
    reader: Reader<R>,
    buffer: Vec<u8>,
    started: bool,
    declared: bool,
}
impl<R: BufRead> Xml<R> {
    fn new(input: R) -> Self {
        let mut reader = Reader::from_reader(input);
        reader.config_mut().expand_empty_elements = true;
        reader.config_mut().check_comments = true;
        Self {
            reader,
            buffer: Vec::new(),
            started: false,
            declared: false,
        }
    }
    fn next(&mut self) -> Result<Token> {
        loop {
            self.buffer.clear();
            match self.reader.read_event_into(&mut self.buffer)? {
                Event::Start(node) => {
                    self.started = true;
                    // Validate attributes even though this schema does not use them.
                    for attribute in node.attributes() {
                        attribute?.decode_and_unescape_value(self.reader.decoder())?;
                    }
                    return Ok(Token::Open(
                        std::str::from_utf8(node.name().as_ref())?.to_owned(),
                    ));
                }
                Event::End(_) => return Ok(Token::Close),
                Event::Text(text) => return Ok(Token::Text(text.unescape()?.into_owned())),
                Event::CData(text) => return Ok(Token::Text(text.decode()?.into_owned())),
                Event::Decl(_) => {
                    require(!self.started && !self.declared, "misplaced XML declaration")?;
                    self.declared = true;
                }
                Event::Comment(_) | Event::PI(_) => {}
                Event::DocType(_) => return Err("XML document types are not supported".into()),
                Event::Eof => return Ok(Token::Eof),
                Event::Empty(_) => unreachable!("empty elements are expanded"),
            }
        }
    }
    fn child(&mut self) -> Result<Option<String>> {
        loop {
            match self.next()? {
                Token::Open(name) => return Ok(Some(name)),
                Token::Close => return Ok(None),
                Token::Text(text) => require(text.trim().is_empty(), "unexpected XML text")?,
                Token::Eof => return Err("unexpected end of XML".into()),
            }
        }
    }
    fn scalar(&mut self) -> Result<String> {
        let mut text = String::new();
        loop {
            match self.next()? {
                Token::Text(part) => {
                    if text.is_empty() {
                        text = part;
                    } else {
                        text.push_str(&part);
                    }
                }
                Token::Close => {
                    let start = text.len() - text.trim_start().len();
                    let len = text.trim().len();
                    text.drain(..start);
                    text.truncate(len);
                    return Ok(text);
                }
                _ => return Err("expected scalar XML text".into()),
            }
        }
    }
    fn vector(&mut self, tag: &str) -> Result<Vec<String>> {
        let mut values = Vec::new();
        while let Some(name) = self.child()? {
            require(name == tag, &format!("expected <{tag}> children"))?;
            values.push(self.scalar()?);
        }
        Ok(values)
    }
    fn polynomials(&mut self) -> Result<Vec<Vec<String>>> {
        let mut polys = Vec::new();
        while let Some(name) = self.child()? {
            require(name == "polynomial", "expected <polynomial>")?;
            polys.push(self.vector("coeff")?);
        }
        Ok(polys)
    }
    fn matrix(&mut self) -> Result<PolynomialMatrix> {
        let (mut rows, mut cols, mut elements) = (None, None, None);
        let mut matrix = PolynomialMatrix::default();
        let mut seen = std::collections::HashSet::new();
        while let Some(name) = self.child()? {
            require(seen.insert(name.clone()), "duplicate XML matrix field")?;
            match name.as_str() {
                "rows" => rows = Some(self.scalar()?.parse::<usize>()?),
                "cols" => cols = Some(self.scalar()?.parse::<usize>()?),
                "elements" => {
                    let mut vectors = Vec::new();
                    while let Some(tag) = self.child()? {
                        require(tag == "polynomialVector", "expected <polynomialVector>")?;
                        vectors.push(self.polynomials()?);
                    }
                    elements = Some(vectors);
                }
                "samplePoints" => matrix.sample_points = Some(self.vector("elt")?),
                "sampleScalings" => matrix.sample_scalings = Some(self.vector("elt")?),
                "bilinearBasis" => matrix.bilinear_basis = Some(self.polynomials()?),
                _ => return Err(format!("unknown XML matrix field <{name}>").into()),
            }
        }
        let rows = rows.ok_or("missing <rows>")?;
        let cols = cols.ok_or("missing <cols>")?;
        let elements = elements.ok_or("missing <elements>")?;
        require(
            rows > 0 && rows == cols,
            "XML polynomial matrix must be nonempty and square",
        )?;
        require(
            rows.checked_mul(cols) == Some(elements.len()),
            "XML element count does not match rows and cols",
        )?;
        let mut elements = elements.into_iter();
        matrix.polynomials = (0..rows)
            .map(|_| elements.by_ref().take(cols).collect())
            .collect();
        Ok(matrix)
    }
}
pub(crate) fn scan(
    input: impl BufRead,
    mut emit: Option<&mut dyn FnMut(PolynomialMatrix) -> Result<()>>,
) -> Result<crate::stream::Header> {
    let mut xml = Xml::new(input);
    require(
        xml.child()?.as_deref() == Some("sdp"),
        "expected <sdp> root",
    )?;
    let (mut objective, mut matrices) = (None, None);
    while let Some(name) = xml.child()? {
        match name.as_str() {
            "objective" => {
                require(objective.is_none(), "duplicate <objective>")?;
                objective = Some(xml.vector("elt")?);
            }
            "polynomialVectorMatrices" => {
                require(matrices.is_none(), "duplicate <polynomialVectorMatrices>")?;
                let mut count = 0usize;
                while let Some(tag) = xml.child()? {
                    require(
                        tag == "polynomialVectorMatrix",
                        "expected <polynomialVectorMatrix>",
                    )?;
                    if let Some(emit) = &mut emit {
                        emit(xml.matrix()?)?;
                    } else {
                        // First pass needs only the header and block count.
                        // The conversion pass validates all matrix fields.
                        xml.reader.read_to_end_into(
                            quick_xml::name::QName(b"polynomialVectorMatrix"),
                            &mut xml.buffer,
                        )?;
                    }
                    count = count.checked_add(1).ok_or("too many matrices")?;
                }
                matrices = Some(count);
            }
            _ => return Err(format!("unknown XML field <{name}>").into()),
        }
    }
    loop {
        match xml.next()? {
            Token::Eof => break,
            Token::Text(text) if text.trim().is_empty() => {}
            _ => return Err("content after XML root".into()),
        }
    }
    Ok(crate::stream::Header {
        objective: objective.ok_or("missing <objective>")?,
        normalization: None,
        count: matrices.ok_or("missing <polynomialVectorMatrices>")?,
    })
}

pub(crate) fn read(input: impl BufRead) -> Result<PolynomialMatrixProgram> {
    let mut matrices = Vec::new();
    let header = scan(
        input,
        Some(&mut |matrix| {
            matrices.push(matrix);
            Ok(())
        }),
    )?;
    Ok(PolynomialMatrixProgram {
        objective: header.objective,
        normalization: None,
        matrices,
    })
}
